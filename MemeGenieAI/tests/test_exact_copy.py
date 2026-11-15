import json
from unittest.mock import MagicMock,patch
import pytest
from PIL import Image
from ai_meme_layout import suggest_layout,requested_exact_captions


@pytest.mark.parametrize('alternatives',[False,True])
def test_noncompliant_model_cannot_replace_exact_user_copy(alternatives):
    layer=dict(text='The model ignored the brief.',x=.05,y=.05,width=.9,size=24,align='left',font_style='bold',color='white')
    payload=dict(image_description='A neutral wall.',layers=[layer])
    if alternatives:
        payload=dict(image_description='A neutral wall.',protected_regions=[],suggestions=[
            dict(name='Bold',layers=[layer]),dict(name='Serif',layers=[{**layer,'text':'Another unwanted rewrite.','font_style':'serif'}])])
    session=MagicMock();session.__enter__.return_value=session
    session.get.return_value.json.return_value={'models':[{'name':'gemma3:4b','capabilities':['vision']}]}
    session.post.return_value.content=b'{}';session.post.return_value.json.return_value={'message':{'content':json.dumps(payload)}}
    exact="My solo has a copay.\nNo refunds!"
    with patch('ai_meme_layout.requests.Session',return_value=session):
        _,result=suggest_layout(Image.new('RGB',(1080,1920),'#444444'),f'Use these words exactly: "{exact}"',alternatives=alternatives)
    groups=[o['layers'] for o in result] if alternatives else [result]
    assert all([layer.text for layer in layers]==[exact] for layers in groups)
    assert session.post.call_count==1  # No model rewrite of literal user copy.


def test_two_exact_layers_survive_missing_model_layer():
    from ai_meme_layout import preserve_exact_captions
    from meme_project import TextLayer
    instruction='Use two captions exactly: top "SETUP" and bottom "PAYOFF"'
    result=preserve_exact_captions([{'layers':[TextLayer(text='Wrong')]}],requested_exact_captions(instruction))
    assert [l.text for l in result[0]['layers']]==['SETUP','PAYOFF']
    assert result[0]['layers'][0].y<result[0]['layers'][1].y


@pytest.mark.parametrize('instruction',["A meme about not exactly having a 'great' day",'Explain exactly how Mondays feel'])
def test_ordinary_brief_is_not_literal_copy(instruction):
    assert requested_exact_captions(instruction) is None


@pytest.mark.parametrize('instruction',['Exactly: Hello world','Verbatim: Hello world','Use these exact words: Hello world','Use exactly "hello" in "bold"','Use exactly ""'])
def test_ambiguous_exact_copy_is_rejected_before_inference(instruction):
    with patch('ai_meme_layout.requests.Session') as session,pytest.raises(ValueError):
        suggest_layout(Image.new('RGB',(512,512)),instruction)
    session.assert_not_called()
