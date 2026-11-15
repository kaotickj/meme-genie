import copy
from unittest.mock import patch
import pytest
from ai_meme_layout import validate_suggestion

VALID={'image_description':'A red circle and blue square on a light background.', 'layers':[{'text':'Trying to fit in','x':.05,'y':.05,'width':.9,'size':36,'align':'center','font_style':'bold','color':'white'}]}

def test_valid_caption_data():
    description,layers=validate_suggestion(VALID)
    assert layers[0].text=='Trying to fit in'
    assert layers[0].font=='DejaVuSans-Bold.ttf'

@pytest.mark.parametrize('key,value',[('x',float('nan')),('y',101),('width',200),('size',1000),('font_style','../../anything'),('text',''),('color','eval(...)')])
def test_invalid_caption_data_rejected(key,value):
    data=copy.deepcopy(VALID);data['layers'][0][key]=value
    with pytest.raises(ValueError):validate_suggestion(data)

def test_ai_preview_cancel_and_apply(tmp_path):
    from PIL import Image
    from PyQt6.QtWidgets import QApplication,QDialog
    from PyQt6.QtCore import QSettings
    from memegenie_ai import MemeGeneratorApp
    app=QApplication.instance() or QApplication([])
    with patch('memegenie_ai.QSettings',return_value=QSettings(str(tmp_path/'ai.ini'),QSettings.Format.IniFormat)):
        w=MemeGeneratorApp();w.base_image=Image.new('RGBA',(512,512),'gray')
        w._layout_background=w.base_image.copy();w.text_input.setPlainText('Keep me')
        desc,layers=validate_suggestion(VALID)
        with patch.object(QDialog,'exec',return_value=QDialog.DialogCode.Rejected):w.preview_ai_layout(desc,layers)
        assert w.text_input.toPlainText()=='Keep me'
        with patch.object(QDialog,'exec',return_value=QDialog.DialogCode.Accepted):w.preview_ai_layout(desc,layers)
        assert w.text_input.toPlainText()=='Trying to fit in'
        assert not w.auto_layout_checkbox.isChecked()
        assert w.font_size.value()==36 and w.layer_y.value()==5
        w.text_input.setPlainText('My edit');assert w.text_layers[0].text=='My edit'
        w.close()


def test_local_cpu_request_and_missing_vision(tmp_path):
    import json
    from PIL import Image
    from unittest.mock import MagicMock
    from ai_meme_layout import suggest_layout,ENDPOINT
    session=MagicMock();session.__enter__.return_value=session
    session.get.return_value.json.return_value={'models':[{'name':'gemma3:4b','capabilities':['vision']}]}
    response=session.post.return_value
    response.content=b'{}';response.json.return_value={'message':{'content':json.dumps(VALID)}}
    with patch('ai_meme_layout.requests.Session',return_value=session):
        description,layers=suggest_layout(Image.new('RGB',(512,512)),'Test')
    assert session.trust_env is False
    args,kwargs=session.post.call_args
    assert args[0]==ENDPOINT+'/api/chat' and kwargs['allow_redirects'] is False
    assert 'num_gpu' not in kwargs['json']['options'] and 'num_thread' not in kwargs['json']['options']
    assert kwargs['json']['keep_alive']==0
    assert kwargs['json']['messages'][0]['images']
    session.post.reset_mock();session.get.return_value.json.return_value={'models':[]}
    with patch('ai_meme_layout.requests.Session',return_value=session):
        with pytest.raises(ValueError):suggest_layout(Image.new('RGB',(20,20)),'Test')
    session.post.assert_not_called()


def test_ai_near_boundary_and_percent_coordinates_fit():
    data=copy.deepcopy(VALID)
    data['layers'][0].update(x=.25,y=.9,width=.9,size=80)
    _,layers=validate_suggestion(data)
    assert layers[0].x+layers[0].width<=1 and layers[0].y<=.72 and layers[0].size==72
    data['layers'][0].update(x=10,y=20,width=80,size=32)
    _,layers=validate_suggestion(data)
    assert (layers[0].x,layers[0].y,layers[0].width)==(.1,.2,.8)


def test_creative_brief_prompt_distinguishes_caption_from_instruction():
    from ai_meme_layout import build_caption_prompt
    brief='A witty caption about music being cheaper than therapy'
    prompt=build_caption_prompt(brief)
    assert 'creative brief, not text to copy' in prompt
    assert 'do not merely restate the theme' in prompt
    assert 'explicitly asks to use quoted words exactly' in prompt
    assert prompt.endswith('User request: '+brief)


def test_distinct_options_and_choice_apply(tmp_path):
    from ai_meme_layout import validate_options
    from PIL import Image
    from PyQt6.QtWidgets import QApplication,QDialog,QComboBox
    from PyQt6.QtCore import QSettings
    from memegenie_ai import MemeGeneratorApp
    second=copy.deepcopy(VALID['layers']);second[0].update(text='A completely different joke',y=.65,font_style='serif',color='yellow',align='left')
    payload={'image_description':VALID['image_description'],'suggestions':[{'name':'Dry humor','layers':VALID['layers']},{'name':'Punchy headline','layers':second}]}
    desc,options=validate_options(payload)
    app=QApplication.instance() or QApplication([])
    with patch('memegenie_ai.QSettings',return_value=QSettings(str(tmp_path/'choices.ini'),QSettings.Format.IniFormat)):
        w=MemeGeneratorApp();w.base_image=Image.new('RGBA',(512,512),'gray');w._layout_background=w.base_image.copy()
        def choose_second(dialog):
            dialog.findChild(QComboBox).setCurrentIndex(1)
            return QDialog.DialogCode.Accepted
        with patch.object(QDialog,'exec',choose_second):w.preview_ai_layout(desc,options)
        assert w.text_input.toPlainText()=='A completely different joke'
        assert w.font_selector.currentText()=='DejaVuSerif-Bold.ttf'
        assert w.layer_y.value()==65 and w.text_color==(255,224,64,255)
        w.close()
    payload['suggestions'][1]['layers']=copy.deepcopy(payload['suggestions'][0]['layers'])
    with pytest.raises(ValueError,match='repeated'):validate_options(payload)


def test_exact_wording_can_have_different_designs():
    from ai_meme_layout import validate_options
    alternate=copy.deepcopy(VALID['layers']);alternate[0].update(font_style='serif',color='yellow')
    desc,options=validate_options({'image_description':VALID['image_description'],'suggestions':[{'name':'Bold','layers':VALID['layers']},{'name':'Editorial','layers':alternate}]})
    assert options[0]['layers'][0].text==options[1]['layers'][0].text
    assert options[0]['layers'][0].font!=options[1]['layers'][0].font


def test_ai_uses_selected_crop_and_rejects_orientation_change(tmp_path):
    from unittest.mock import MagicMock
    from PIL import Image
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QSettings
    from memegenie_ai import MemeGeneratorApp
    app=QApplication.instance() or QApplication([])
    with patch('memegenie_ai.QSettings',return_value=QSettings(str(tmp_path/'crop.ini'),QSettings.Format.IniFormat)):
        w=MemeGeneratorApp();w.base_image=Image.new('RGBA',(800,600),'gray')
        w.preview_format.setCurrentIndex(2)
        fake=MagicMock();fake.isRunning.return_value=False
        with patch('memegenie_ai.LayoutWorker',return_value=fake) as worker:w.start_ai_layout()
        assert worker.call_args.args[0].size==(1080,1920)
        desc,layers=validate_suggestion(VALID)
        w.preview_format.setCurrentIndex(1)
        w.preview_ai_layout(desc,layers)
        assert 'orientation changed' in w.caption_ai_status.text()
        w.close()


def test_ai_collision_guard_shrinks_upper_caption():
    from PIL import Image
    from meme_project import TextLayer
    from ai_meme_layout import fit_caption_layers
    layers=[TextLayer(text='My therapist says I need to process my feelings.',x=.25,y=.4,width=.75,size=32),TextLayer(text='I just told him to lower the pitch.',x=.25,y=.6,width=.75,size=24)]
    fitted=fit_caption_layers(Image.new('RGB',(512,512)),layers)
    assert fitted[0].size<layers[0].size
    assert fitted[0].text==layers[0].text and fitted[1].y==layers[1].y
    assert layers[0].size==32
