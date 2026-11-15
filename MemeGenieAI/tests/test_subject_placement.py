from dataclasses import asdict
import pytest
from PIL import Image, ImageChops
from ai_meme_layout import validate_regions, place_around_subjects
from meme_project import TextLayer, render_layers


def test_rendered_caption_moves_away_from_subject_and_remains_editable():
    image=Image.new('RGB',(800,450),'#333333')
    layer=TextLayer(text='My solo has a copay.',x=.53,y=.18,width=.42,size=28,font='DejaVuSerif-Bold.ttf')
    before=asdict(layer)
    regions=validate_regions([dict(label='head and guitar',x=.5,y=.02,width=.48,height=.95)])
    placed=place_around_subjects(image,[layer],regions)
    changed=ImageChops.difference(image,render_layers(image,placed).convert('RGB'))
    assert changed.crop((400,0,800,450)).getbbox() is None
    assert changed.getbbox() is not None and placed[0].size>=16
    assert placed[0].text==layer.text and placed[0].font==layer.font
    assert asdict(layer)==before
    placed[0].text='Still editable'


def test_no_subject_free_space_is_reported_not_silently_covered():
    image=Image.new('RGB',(512,512),'gray')
    with pytest.raises(ValueError,match='No readable caption space'):
        place_around_subjects(image,[TextLayer(text='A caption')],[dict(x=0,y=0,width=1,height=1)])


@pytest.mark.parametrize('update',[{'x':float('nan')},{'width':0},{'height':2},{'x':1,'width':.5},{'label':''},{'y':True}])
def test_subject_boxes_reject_invalid_model_data(update):
    region=dict(label='face',x=.1,y=.1,width=.2,height=.2)
    with pytest.raises(ValueError):validate_regions([{**region,**update}])


def test_observed_qwen_grid_is_normalized_without_changing_input():
    source=[dict(label='head',x=173,y=62,width=.15,height=.18)]
    region=validate_regions(source)[0]
    assert (region['x'],region['y'],region['width'],region['height'])==(.173,.062,.15,.18)
    assert source[0]['x']==173
    cropped=validate_regions([dict(label='guitar',x=.8,y=.2,width=.5,height=.4)])[0]
    assert cropped['x']+cropped['width']==1
    with pytest.raises(ValueError):validate_regions([dict(label='head',x=1001,y=0,width=.2,height=.2)])


def test_multiple_captions_do_not_overlap():
    image=Image.new('RGB',(800,450),'#333333')
    layers=[TextLayer(text='First joke',size=28),TextLayer(text='Second joke',size=28)]
    placed=place_around_subjects(image,layers,[])
    masks=[render_layers(Image.new('RGBA',image.size),[l]).getchannel('A') for l in placed]
    from PIL import ImageChops
    assert ImageChops.multiply(*masks).getbbox() is None
    assert masks[0].getbbox()[3]<masks[1].getbbox()[1]


def test_relocated_caption_gets_readable_contrast():
    layer=TextLayer(text='A quiet aside',color=(0,0,0,255),size=24)
    dark=Image.new('RGB',(800,450),'#181818')
    placed=place_around_subjects(dark,[layer],[])
    assert placed[0].color==(255,255,255,255)
    assert layer.color==(0,0,0,255)


def test_local_options_request_protects_subjects_and_preserves_exact_words():
    import json
    from unittest.mock import MagicMock,patch
    from ai_meme_layout import suggest_layout
    record=dict(text='My solo has a copay.',x=.53,y=.18,width=.42,size=28,
                font_style='serif',align='left',color='white')
    payload=dict(image_description='A musician on the right.',
                 protected_regions=[dict(label='musician',x=.5,y=0,width=.5,height=1)],
                 suggestions=[dict(name='Editorial',layers=[record]),
                              dict(name='Deadpan',layers=[{**record,'font_style':'mono'}])])
    session=MagicMock();session.__enter__.return_value=session
    session.get.return_value.json.return_value={'models':[{'name':'gemma3:4b','capabilities':['vision']}]}
    session.post.return_value.content=b'{}'
    session.post.return_value.json.return_value={'message':{'content':json.dumps(payload)}}
    with patch('ai_meme_layout.requests.Session',return_value=session):
        _,options=suggest_layout(Image.new('RGB',(800,450),'gray'),'Use these words exactly: "My solo has a copay."',alternatives=True)
    assert all(o['layers'][0].text==record['text'] for o in options)
    assert all(o['layers'][0].x<.5 for o in options)
    request=session.post.call_args.kwargs['json']
    assert 'protected_regions' in request['format']['required']
    assert request['keep_alive']=='2m'
    assert 'num_gpu' not in request['options'] and 'num_thread' not in request['options'] and session.trust_env is False
    from ai_meme_layout import NoCaptionSpace
    with patch('ai_meme_layout.requests.Session',return_value=session), patch('ai_meme_layout.place_around_subjects',side_effect=[NoCaptionSpace('No room'),[TextLayer(text=record['text'])]]):
        _,surviving=suggest_layout(Image.new('RGB',(800,450),'gray'),'Use these words exactly: "My solo has a copay."',alternatives=True)
    assert len(surviving)==1 and surviving[0]['name']=='Deadpan'


def test_apply_preserves_checked_side_by_side_positions(tmp_path):
    from unittest.mock import patch
    from PyQt6.QtWidgets import QApplication,QDialog
    from PyQt6.QtCore import QSettings
    from memegenie_ai import MemeGeneratorApp
    app=QApplication.instance() or QApplication([])
    layers=[TextLayer(text='One',x=.04,y=.05,width=.4,size=20),
            TextLayer(text='Two',x=.56,y=.05,width=.4,size=20)]
    with patch('memegenie_ai.QSettings',return_value=QSettings(str(tmp_path/'subjects.ini'),QSettings.Format.IniFormat)):
        window=MemeGeneratorApp()
        window.base_image=Image.new('RGBA',(800,450),'gray')
        window._layout_background=window.base_image.copy()
        options=[dict(name='Protected composition',layers=layers,subject_checked=True)]
        with patch.object(QDialog,'exec',return_value=QDialog.DialogCode.Accepted):
            window.preview_ai_layout('Two subjects',options)
        assert [(l.x,l.y) for l in window.text_layers]==[(.04,.05),(.56,.05)]
        window.close()


def test_focused_writing_pass_keeps_styles_and_uses_same_cpu_model():
    import json,time
    from unittest.mock import MagicMock
    from ai_meme_layout import refine_caption_options
    options=[dict(name='Editorial',layers=[TextLayer(text='weak',font='DejaVuSerif-Bold.ttf')]),
             dict(name='Deadpan',layers=[TextLayer(text='also weak',font='DejaVuSansMono-Bold.ttf')])]
    session=MagicMock()
    session.post.return_value.content=b'{}'
    session.post.return_value.json.return_value={'message':{'content':json.dumps({'captions':['Fresh first joke','Fresh second joke']})}}
    result=refine_caption_options(session,'gemma3:4b','A band on stage','Band humor',options,[],time.monotonic()+30,max_chars=60)
    assert [o['layers'][0].text for o in result]==['Fresh first joke','Fresh second joke']
    assert [o['layers'][0].font for o in result]==[o['layers'][0].font for o in options]
    request=session.post.call_args.kwargs
    assert request['json']['model']=='gemma3:4b' and 'num_gpu' not in request['json']['options'] and 'num_thread' not in request['json']['options']
    assert request['json']['keep_alive']==0
    assert request['json']['format']['properties']['captions']['items']['maxLength']>60
    assert request['timeout'][1]<=30 and request['allow_redirects'] is False
    assert options[0]['layers'][0].text=='weak'
    session.reset_mock()
    assert refine_caption_options(session,'gemma3:4b','A band','Use my wording exactly',options,[],time.monotonic()+30)==options
    session.post.assert_not_called()


def test_writing_pass_rejects_repetition_and_expired_deadline():
    import json,time
    from unittest.mock import MagicMock
    from ai_meme_layout import refine_caption_options
    options=[dict(name='One',layers=[TextLayer(text='one')]),dict(name='Two',layers=[TextLayer(text='two')])]
    session=MagicMock();session.post.return_value.content=b'{}'
    session.post.return_value.json.return_value={'message':{'content':json.dumps({'captions':['Same joke','same joke']})}}
    with pytest.raises(ValueError,match='repeated'):
        refine_caption_options(session,'gemma3:4b','A band','Band humor',options,[],time.monotonic()+30)
    session.reset_mock()
    with pytest.raises(ValueError,match='ran out of time'):
        refine_caption_options(session,'gemma3:4b','A band','Band humor',options,[],time.monotonic()-1)
    session.post.assert_not_called()
