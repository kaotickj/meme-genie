"""Offline image-provider tests; no real HTTP or user images."""
import base64
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
from PIL import Image, ImageChops
import openai_images as cloud


@pytest.fixture
def transport(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-key')
    session = MagicMock(); session.__enter__.return_value = session
    response = session.post.return_value.__enter__.return_value
    response.status_code = 200
    image = Image.new('RGB', (1024,1024), '#334455')
    stream = io.BytesIO(); image.save(stream, format='PNG')
    payload = {'data':[{'b64_json':base64.b64encode(stream.getvalue()).decode()}]}
    response.iter_content.side_effect = lambda _: [json.dumps(payload).encode()]
    monkeypatch.setattr(cloud.requests, 'Session', lambda: session)
    return session, response, payload


def test_text_only_generation_contract_and_resolution(transport):
    session,_,_ = transport
    image = cloud.generate_image('A blue sky', 'letters', (512,512))
    assert image.size == (512,512) and image.mode == 'RGBA'
    args, kw = session.post.call_args
    assert args == (cloud.ENDPOINT,)
    assert kw['json'] == {'model':cloud.MODEL,'prompt':'A blue sky\n\nAvoid these elements where possible: letters',
        'n':1,'size':'1024x1024','quality':'medium','complexity':'medium','output_format':'png','background':'opaque'}
    assert kw['allow_redirects'] is False and session.trust_env is False
    assert session.post.call_count == 1 and not session.get.called


@pytest.mark.parametrize('size,native', [((512,512),'1024x1024'),((1024,576),'1536x1024'),((576,1024),'1024x1536')])
def test_supported_sizes(size,native):
    assert cloud.native_size(size) == native


@pytest.mark.parametrize('prompt,negative,size', [('', '', (512,512)), ('x'*32001,'',(512,512)), ('ok',None,(512,512)), ('ok','',(0,10)), ('ok','',(True,512))])
def test_validation_before_network(prompt,negative,size):
    with patch.object(cloud.requests, 'Session') as session:
        with pytest.raises(ValueError): cloud.generate_image(prompt,negative,size)
        session.assert_not_called()


def test_missing_key_before_network(monkeypatch,tmp_path):
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    monkeypatch.setenv('MEMEGENIE_OPENAI_KEY_FILE',str(tmp_path/'missing'))
    with patch.object(cloud.requests,'Session') as session:
        with pytest.raises(ValueError,match='OPENAI_API_KEY'):cloud.generate_image('sky','',(512,512))
        session.assert_not_called()


@pytest.mark.parametrize('status',[302,400,401,403,404,429,500])
def test_http_error_no_retry(transport,status):
    session,response,_=transport;response.status_code=status
    with pytest.raises(ValueError,match=f'HTTP {status}') as error:cloud.generate_image('sky','',(512,512))
    assert 'synthetic-key' not in str(error.value) and session.post.call_count==1


@pytest.mark.parametrize('error',[requests.Timeout('SECRET'),requests.ConnectionError('SECRET')])
def test_network_errors_sanitized(transport,error):
    session,_,_=transport;session.post.side_effect=error
    with pytest.raises(ValueError) as result:cloud.generate_image('sky','',(512,512))
    assert 'SECRET' not in str(result.value) and 'charged' in str(result.value)


@pytest.mark.parametrize('variant',['missing','url','invalid_base64','not_image','wrong_size','json','oversize','refusal'])
def test_invalid_responses(transport,variant):
    _,response,payload=transport
    if variant=='missing':payload['data']=[]
    if variant=='url':payload['data']=[{'url':'https://untrusted.invalid/image'}]
    if variant=='invalid_base64':payload['data'][0]['b64_json']='!!!'
    if variant=='not_image':payload['data'][0]['b64_json']=base64.b64encode(b'not png').decode()
    if variant=='wrong_size':
        buf=io.BytesIO();Image.new('RGB',(20,20)).save(buf,format='PNG')
        payload['data'][0]['b64_json']=base64.b64encode(buf.getvalue()).decode()
    if variant=='json':response.iter_content.side_effect=lambda _:[b'bad json']
    if variant=='oversize':response.iter_content.side_effect=lambda _:[b'x'*(cloud.MAX_RESPONSE+1)]
    if variant=='refusal':payload['error']={'message':'SECRET'}
    with pytest.raises(ValueError) as result:cloud.generate_image('sky','',(512,512))
    assert 'SECRET' not in str(result.value)


def test_cancel_before_and_after_submission(transport):
    session,_,_=transport
    with pytest.raises(ValueError,match='before sending'):cloud.generate_image('sky','',(512,512),canceled=lambda:True)
    session.post.assert_not_called()
    checks=iter([False,True])
    with pytest.raises(ValueError,match='discarded'):cloud.generate_image('sky','',(512,512),canceled=lambda:next(checks))
    assert session.post.call_count==1


@pytest.fixture
def window(tmp_path):
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QSettings
    import memegenie_ai as gui
    app=QApplication.instance() or QApplication([])
    with patch.object(gui,'QSettings',return_value=QSettings(str(tmp_path/'settings.ini'),QSettings.Format.IniFormat)):
        w=gui.MemeGeneratorApp();w.output_dir.setText(str(tmp_path));yield w
        if w.image_worker: w.image_worker = None
        w.close()


def test_gui_provider_routing_and_preserved_controls(window,transport):
    import memegenie_ai as gui
    assert window.image_provider.currentData()=='local'
    window.ai_seed.setValue(123);window.ai_steps.setValue(27)
    window.prompt_preset.setCurrentIndex(1)
    prompt=window.ai_prompt.toPlainText();assert prompt
    window.image_provider.setCurrentIndex(1)
    assert not window.ai_seed.isEnabled() and 'paid' in window.image_provider_note.text()
    with patch.object(gui.OpenAIImageWorker,'start'),patch.object(gui,'build_sd_cli_arguments') as local:
        window.generate_ai_image(); local.assert_not_called()
    assert window.image_worker.prompt==prompt
    window.image_provider.setCurrentIndex(0)
    assert window.ai_seed.isEnabled() and window.ai_seed.value()==123 and window.ai_steps.value()==27
    with patch.object(gui,'build_sd_cli_arguments',side_effect=ValueError('local sentinel')) as local,patch.object(gui.QMessageBox,'warning'),patch.object(window,'start_openai_image') as online:
        window.generate_ai_image();local.assert_called_once();online.assert_not_called()
        assert local.call_args.args[0].seed==123
    # Caption provider remains independent and locally selected.
    assert window.caption_model.currentText()=='gemma3:4b'


def test_success_saves_credit_opens_clean_source_and_exports(window,tmp_path):
    import memegenie_ai as gui
    from PIL import ImageDraw
    image=Image.new('RGBA',(512,512),'#334455');before=image.tobytes()
    output=tmp_path/'generated.png'
    window.image_worker=gui.OpenAIImageWorker('sky','letters',(512,512),output)
    with patch.object(ImageDraw.ImageDraw,'text',autospec=True,wraps=ImageDraw.ImageDraw.text) as draw:
        # Separate explicit credit-text check; actual render below uses real draw.
        window.add_attribution(image)
        assert any('https://github.com/kaotickj' in str(call) for call in draw.call_args_list)
    window.openai_image_ready(image)
    assert window.base_image.tobytes()==before and image.tobytes()==before
    assert window.tabs.currentIndex()==0 and output.is_file()
    with Image.open(output) as saved:
        assert saved.convert('RGBA').tobytes()==window.add_attribution(image).tobytes()
        assert saved.info['Meme Genie AI model']==cloud.MODEL
    exported=tmp_path/'edited.png'
    with patch.object(gui.QFileDialog,'getSaveFileName',return_value=(str(exported),'PNG')),patch.object(gui.QMessageBox,'information'):
        window.generate_meme()
    with Image.open(exported) as saved:
        assert saved.convert('RGBA').tobytes()==window.render_meme().tobytes()
    assert not list(tmp_path.glob('.memegenie-*'))


def test_cancel_and_save_failure_preserve_editor(window,tmp_path):
    import memegenie_ai as gui
    window.base_image=Image.new('RGBA',(512,512),'red');before=window.base_image.tobytes()
    target=tmp_path/'result.png';window.image_worker=gui.OpenAIImageWorker('sky','',(512,512),target)
    window.ai_canceled=True;window.openai_image_ready(Image.new('RGBA',(512,512),'blue'))
    assert window.base_image.tobytes()==before and not target.exists()
    window.ai_canceled=False
    with patch.object(gui.os,'replace',side_effect=OSError('test failure')):
        window.openai_image_ready(Image.new('RGBA',(512,512),'blue'))
    assert window.base_image.tobytes()==before and not target.exists()
    assert not list(tmp_path.glob('.memegenie-*'))


def test_worker_uses_text_only_contract():
    import memegenie_ai as gui
    worker=gui.OpenAIImageWorker('sky','letters',(512,512),Path('unused.png'))
    with patch.object(cloud,'generate_image',return_value=Image.new('RGBA',(512,512))) as generate:
        worker.run()
        assert generate.call_args.args==('sky','letters',(512,512))
