import base64
import io
import json
from unittest.mock import MagicMock, patch
import pytest
import requests
from PIL import Image
import ai_meme_layout as local
import openai_layout as cloud

@pytest.fixture
def transport(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-test-key')
    session = MagicMock(); session.__enter__.return_value = session
    response = session.post.return_value.__enter__.return_value
    response.status_code = 200
    layer = dict(text='My deadline needs a vacation.', x=.05, y=.05, width=.9, size=24, align='left', font_style='bold', color='white')
    data = dict(image_description='A synthetic gray background.', protected_regions=[], suggestions=[dict(name='Bold', layers=[layer]),dict(name='Aside', layers=[{**layer, 'text':'The coffee filed for overtime.', 'font_style':'serif'}])])
    payload = dict(status='completed', output=[dict(type='message', content=[dict(type='output_text', text=json.dumps(data))])])
    response.iter_content.side_effect = lambda _: [json.dumps(payload).encode()]
    monkeypatch.setattr(cloud.requests, 'Session', lambda: session)
    return session, response, payload

def test_luna_contract(transport):
    session, _, _ = transport
    image = Image.new('RGB', (1920,1080), 'gray'); before=image.tobytes()
    description, options = local.suggest_layout(image, 'Make a work joke', 'gpt-6-luna', provider='openai', alternatives=True)
    assert description and len(options)==2 and all(o['subject_checked'] for o in options)
    assert image.tobytes()==before and not session.get.called and session.post.call_count==1
    args, kw=session.post.call_args; body=kw['json']
    assert args==(cloud.ENDPOINT,) and body['model']=='gpt-6-luna' and body['store'] is False
    assert body['text']['format']['strict'] is True and not kw['allow_redirects'] and session.trust_env is False
    encoded=body['input'][0]['content'][1]['image_url'].split(',')[1]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as uploaded: assert uploaded.size==(768,432)
    assert kw['timeout']==(5,120)

def test_local_never_calls_cloud(monkeypatch):
    with patch.object(cloud,'suggest_layout') as online, patch.object(cloud,'api_key') as key:
        session=MagicMock();session.__enter__.return_value=session
        session.get.side_effect=requests.ConnectionError('local unavailable')
        monkeypatch.setattr(local.requests,'Session',lambda:session)
        with pytest.raises(requests.ConnectionError): local.suggest_layout(Image.new('RGB',(20,20)),'test')
        online.assert_not_called(); key.assert_not_called(); session.post.assert_not_called()
        assert session.get.call_args.args[0]==local.ENDPOINT+'/api/tags'

def test_bad_route_no_network():
    with patch.object(cloud.requests,'Session') as session:
        for provider,model in [('unknown','gemma3:4b'),('openai','wrong')]:
            with pytest.raises(ValueError):local.suggest_layout(None,'',model,provider=provider)
        session.assert_not_called()

def test_missing_key_no_network(monkeypatch,tmp_path):
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    monkeypatch.setenv('MEMEGENIE_OPENAI_KEY_FILE',str(tmp_path/'missing'))
    with patch.object(cloud.requests,'Session') as session:
        with pytest.raises(ValueError,match='Set OPENAI_API_KEY'):local.suggest_layout(None,'','gpt-6-luna',provider='openai')
        session.assert_not_called()

def test_private_key(monkeypatch,tmp_path):
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    path=tmp_path/'key';path.write_text('synthetic-test-key\n');path.chmod(0o600)
    monkeypatch.setenv('MEMEGENIE_OPENAI_KEY_FILE',str(path))
    assert cloud.api_key()=='synthetic-test-key'
    if cloud.os.name=='posix':
        path.chmod(0o644)
        with pytest.raises(ValueError,match='private'):cloud.api_key()
    monkeypatch.setenv('OPENAI_API_KEY','environment-test-key')
    assert cloud.api_key()=='environment-test-key'

@pytest.mark.parametrize('status',[301,400,401,403,404,429,500])
def test_http_errors_no_retry(transport,status):
    session,response,_=transport;response.status_code=status
    with pytest.raises(ValueError,match=f'HTTP {status}') as error:cloud.suggest_layout(Image.new('RGB',(32,32)),'',alternatives=True)
    assert 'synthetic-test-key' not in str(error.value) and session.post.call_count==1
    response.iter_content.assert_not_called()

@pytest.mark.parametrize('error',[requests.Timeout('SECRET'),requests.ConnectionError('SECRET')])
def test_network_errors_sanitized(transport,error):
    session,_,_=transport;session.post.side_effect=error
    with pytest.raises(ValueError) as result:cloud.suggest_layout(Image.new('RGB',(32,32)),'')
    assert 'SECRET' not in str(result.value) and session.post.call_count==1

@pytest.mark.parametrize('variant',['incomplete','refusal','missing','bad_json','bad_layout','oversize'])
def test_unusable_output(transport,variant):
    _,response,payload=transport
    if variant=='incomplete':payload['status']='incomplete'
    if variant=='refusal':payload['output'][0]['content']=[{'type':'refusal','refusal':'SECRET'}]
    if variant=='missing':payload['output']=[]
    if variant=='bad_json':response.iter_content.side_effect=lambda _:[b'not json']
    if variant=='bad_layout':payload['output'][0]['content'][0]['text']='{}'
    if variant=='oversize':response.iter_content.side_effect=lambda _:[b'x'*1_000_001]
    with pytest.raises(ValueError) as result:cloud.suggest_layout(Image.new('RGB',(32,32)),'',alternatives=True)
    assert 'SECRET' not in str(result.value)

def test_exact_copy_and_recheck(transport):
    session,_,_=transport
    for recheck in (False,True):
        _,options=local.suggest_layout(Image.new('RGB',(512,512)),'Use exactly "My solo has a copay."','gpt-6-luna',provider='openai',alternatives=True,force_analysis=recheck)
        assert all(o['layers'][0].text=='My solo has a copay.' for o in options)
    assert session.post.call_count==2

def test_worker_and_gui_default(tmp_path):
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QSettings
    import memegenie_ai as gui
    app=QApplication.instance() or QApplication([])
    settings=QSettings(str(tmp_path/'settings.ini'),QSettings.Format.IniFormat)
    with patch.object(gui,'QSettings',return_value=settings):
        window=gui.MemeGeneratorApp()
        assert window.caption_model.currentText()=='gemma3:4b'
        assert [window.caption_model.itemText(i) for i in range(3)]==['gemma3:4b','qwen3.5:9b-q8_0','gemma4:12b-it-q8_0']
        window.base_image=Image.new('RGBA',(512,512),'gray');window.caption_model.setCurrentIndex(3)
        assert 'charges' in window.caption_provider_note.text()
        with patch.object(gui.LayoutWorker,'start'):window.start_ai_layout()
        worker=window.layout_worker
        assert worker.provider=='openai' and worker.model=='gpt-6-luna'
        with patch.object(gui,'suggest_layout',return_value=('scene',[])) as suggest:
            # Avoid opening the modal review dialog during this routing test.
            worker.succeeded.disconnect();worker.run()
            assert suggest.call_args.kwargs['provider']=='openai'
        window.caption_model.setCurrentIndex(0)
        with patch.object(gui.LayoutWorker,'start'):window.start_ai_layout()
        assert window.layout_worker.provider=='local'
        window.close()
