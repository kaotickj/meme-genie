import json
from unittest.mock import MagicMock

import pytest
from PIL import Image
import ai_meme_layout as ai


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr(ai,'_analysis_cache',ai._AnalysisCache())
    monkeypatch.setattr(ai,'place_around_subjects',lambda image,layers,regions:layers)
    state={'images':0,'writing':0,'digest':'weights-v1','model':'gemma3:4b','bad_scene':False,'bad_writing':False}
    session=MagicMock();session.__enter__.return_value=session
    session.get.return_value.json.side_effect=lambda:{'models':[{'name':state['model'],'digest':state['digest'],'capabilities':['vision']}]}
    def post(url,**kwargs):
        request=kwargs['json'];response=MagicMock();response.content=b'{}'
        if request['messages'][0].get('images'):
            state['images']+=1
            layer=dict(text='Draft',x=.05,y=.05,width=.9,size=24,align='left',font_style='bold',color='white')
            payload=dict(image_description='A plain background.',protected_regions=[],suggestions=[
                dict(name='Bold',layers=[layer]),dict(name='Editorial',layers=[{**layer,'font_style':'serif'}])])
            if state['bad_scene']:payload['protected_regions']=[dict(label='bad',x=-1,y=0,width=.1,height=.1)]
        else:
            state['writing']+=1
            payload={'captions':['New joke '+str(state['writing']),'Another joke '+str(state['writing'])]}
            if state['bad_writing']:payload={'captions':['Only one']}
        response.json.return_value={'message':{'content':json.dumps(payload)}}
        return response
    session.post.side_effect=post
    monkeypatch.setattr(ai.requests,'Session',lambda:session)
    return state


def test_repeat_skips_image_analysis_but_writes_fresh_editable_captions(service):
    image=Image.new('RGB',(512,512),'gray')
    _,first=ai.suggest_layout(image,'A funny caption',alternatives=True)
    first[0]['layers'][0].font='user-edit.ttf'
    _,second=ai.suggest_layout(image,'A funny caption',alternatives=True)
    assert service['images']==1 and service['writing']==2
    assert second[0]['layers'][0].text=='New joke 2'
    assert second[0]['layers'][0].font=='DejaVuSans-Bold.ttf'


@pytest.mark.parametrize('change',['pixels','crop','brief','weights','model'])
def test_changed_inputs_require_fresh_image_analysis(service,change):
    image=Image.new('RGB',(512,512),'gray');brief='A funny caption';model='gemma3:4b'
    ai.suggest_layout(image,brief,model=model,alternatives=True)
    if change=='pixels':image.putpixel((0,0),(1,2,3))
    if change=='crop':image=image.crop((0,0,512,400))
    if change=='brief':brief='A different topic'
    if change=='weights':service['digest']='weights-v2'
    if change=='model':service['model']=model='another-vision-model'
    ai.suggest_layout(image,brief,model=model,alternatives=True)
    assert service['images']==2 and service['writing']==2


def test_exact_wording_and_unknown_model_digest_are_not_cached(service):
    image=Image.new('RGB',(512,512),'gray')
    for _ in range(2):ai.suggest_layout(image,'Use exactly "Draft".',alternatives=True)
    assert service['images']==2 and service['writing']==0
    service['digest']=None
    for _ in range(2):ai.suggest_layout(image,'A funny caption',alternatives=True)
    assert service['images']==4 and service['writing']==2


def test_forced_recheck_discards_cached_observations_even_if_it_fails(service):
    image=Image.new('RGB',(512,512),'gray')
    ai.suggest_layout(image,'A funny caption',alternatives=True)
    service['bad_scene']=True
    with pytest.raises(ValueError):
        ai.suggest_layout(image,'A funny caption',alternatives=True,force_analysis=True)
    service['bad_scene']=False
    ai.suggest_layout(image,'A funny caption',alternatives=True)
    assert service['images']==3


def test_bad_scene_is_not_cached_but_valid_analysis_survives_writer_failure(service):
    image=Image.new('RGB',(512,512),'gray')
    service['bad_scene']=True
    with pytest.raises(ValueError):ai.suggest_layout(image,'A funny caption',alternatives=True)
    service['bad_scene']=False;service['bad_writing']=True
    with pytest.raises(ValueError):ai.suggest_layout(image,'A funny caption',alternatives=True)
    service['bad_writing']=False
    ai.suggest_layout(image,'A funny caption',alternatives=True)
    assert service['images']==2 and service['writing']==2


def test_cache_expires_evicts_and_does_not_share_mutable_results(monkeypatch):
    clock=[0];monkeypatch.setattr(ai.time,'monotonic',lambda:clock[0])
    cache=ai._AnalysisCache(capacity=2,ttl=10)
    value={'layers':['original']};cache.put('a',value)
    value['layers'][0]='edited';cached=cache.get('a');cached['layers'][0]='edited again'
    assert cache.get('a')['layers']==['original']
    cache.put('b',{});cache.get('a');cache.put('c',{})
    assert cache.get('b') is None
    clock[0]=11
    assert cache.get('a') is None


def test_boundary_search_finds_space_missed_by_the_old_grid():
    from meme_project import TextLayer,render_layers
    image=Image.new('RGB',(1920,1080),'gray')
    regions=ai.validate_regions([
        dict(label='head of singer',x=.13,y=.08,width=.25,height=.35),
        dict(label='head of guitarist',x=.79,y=.11,width=.24,height=.36),
        dict(label='drummer head',x=.38,y=.37,width=.19,height=.25),
        dict(label='guitar',x=.65,y=.48,width=.4,height=.35)])
    layer=TextLayer(text='That guitar solo cured my bad day for $40 instead of $200',x=.1,y=.72,width=.8,size=42)
    placed=ai.place_around_subjects(image,[layer],regions)
    mask=render_layers(Image.new('RGBA',image.size),placed)
    bounds=mask.getbbox()
    assert bounds is not None and bounds[3]<image.height and placed[0].size>=16
    for region in regions:
        box=tuple(round(v*s) for v,s in zip((region['x'],region['y'],region['x']+region['width'],region['y']+region['height']),(*image.size,*image.size)))
        assert mask.crop(box).getbbox() is None
    assert placed[0].text==layer.text and layer.y==.72


def test_recheck_button_requests_fresh_analysis_without_touching_text(tmp_path):
    from unittest.mock import patch
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QSettings
    from memegenie_ai import MemeGeneratorApp,LayoutWorker
    app=QApplication.instance() or QApplication([])
    with patch('memegenie_ai.QSettings',return_value=QSettings(str(tmp_path/'recheck.ini'),QSettings.Format.IniFormat)):
        window=MemeGeneratorApp();window.base_image=Image.new('RGBA',(512,512),'gray')
        window.text_input.setPlainText('Keep my edit')
        worker=MagicMock();worker.isRunning.return_value=False
        with patch('memegenie_ai.LayoutWorker',return_value=worker) as make_worker:
            window.caption_recheck_button.click()
        assert make_worker.call_args.kwargs['force_analysis'] is True
        assert window.text_input.toPlainText()=='Keep my edit'
        assert not window.caption_ai_button.isEnabled() and not window.caption_recheck_button.isEnabled()
        worker.finished.connect.call_args.args[0]()
        assert window.caption_ai_button.isEnabled() and window.caption_recheck_button.isEnabled()
        with patch('memegenie_ai.LayoutWorker',return_value=worker) as standard:
            window.caption_ai_button.click()
        assert standard.call_args.kwargs['force_analysis'] is False
        window.close()
    real_worker=LayoutWorker(Image.new('RGB',(10,10)),'Brief','gemma3:4b',force_analysis=True)
    with patch('memegenie_ai.suggest_layout',return_value=('Description',[])) as suggest:
        real_worker.run()
    assert suggest.call_args.kwargs['force_analysis'] is True


def test_caption_repair_finishes_fragments_without_rewriting_valid_alternative():
    import time
    from meme_project import TextLayer
    options=[dict(name='One',layers=[TextLayer(text='draft')]),dict(name='Two',layers=[TextLayer(text='draft')])]
    session=MagicMock()
    def response(captions):
        value=MagicMock();value.content=b'{}'
        value.json.return_value={'message':{'content':json.dumps({'captions':captions})}}
        return value
    valid='Band practice is my unpaid group therapy.'
    session.post.side_effect=[response(['My therapist charges more for',valid]),response(['My therapist said open up. The band said turn down.'])]
    result=ai.refine_caption_options(session,'gemma3:4b','A band','Music and therapy',options,[],time.monotonic()+30,max_chars=60)
    assert result[1]['layers'][0].text==valid
    assert result[0]['layers'][0].text.endswith('turn down.')
    assert session.post.call_count==2
    assert all('num_gpu' not in call.kwargs['json']['options'] and 'num_thread' not in call.kwargs['json']['options'] and call.kwargs['json']['keep_alive']==0 for call in session.post.call_args_list)
    assert session.post.call_args.kwargs['json']['format']['properties']['captions']['minItems']==1


def test_caption_repair_is_bounded_and_preserves_original_layers():
    import time
    from meme_project import TextLayer
    options=[dict(name='One',layers=[TextLayer(text='keep')]),dict(name='Two',layers=[TextLayer(text='also keep')])]
    session=MagicMock();session.post.return_value.content=b'{}'
    session.post.return_value.json.return_value={'message':{'content':json.dumps({'captions':['A caption that stops at','A caption that ends with']})}}
    with pytest.raises(ValueError,match='could not finish'):
        ai.refine_caption_options(session,'gemma3:4b','A band','Music and therapy',options,[],time.monotonic()+30,max_chars=60)
    assert session.post.call_count==2 and options[0]['layers'][0].text=='keep'


def test_repair_cannot_start_after_the_shared_deadline(monkeypatch):
    from meme_project import TextLayer
    clock=[0];monkeypatch.setattr(ai.time,'monotonic',lambda:clock[0])
    options=[dict(name='One',layers=[TextLayer()]),dict(name='Two',layers=[TextLayer()])]
    session=MagicMock()
    def slow_response(*args,**kwargs):
        clock[0]=11
        response=MagicMock();response.content=b'{}'
        response.json.return_value={'message':{'content':json.dumps({'captions':['Ends with','Ends at']})}}
        return response
    session.post.side_effect=slow_response
    with pytest.raises(ValueError,match='ran out of time'):
        ai.refine_caption_options(session,'gemma3:4b','A band','Music and therapy',options,[],10,max_chars=60)
    assert session.post.call_count==1


def test_failed_shortening_does_not_discard_another_valid_joke():
    import time
    from meme_project import TextLayer
    options=[dict(name='Keep this',layers=[TextLayer()]),dict(name='Too long',layers=[TextLayer()])]
    session=MagicMock()
    def response(captions):
        result=MagicMock();result.content=b'{}'
        result.json.return_value={'message':{'content':json.dumps({'captions':captions})}}
        return result
    session.post.side_effect=[response(['My group therapy has a drummer.','This ends with']),response(['Still ends with'])]
    result=ai.refine_caption_options(session,'gemma3:4b','A band','Music and therapy',options,[],time.monotonic()+30,max_chars=60)
    assert len(result)==1 and result[0]['name']=='Keep this'
    assert result[0]['layers'][0].text=='My group therapy has a drummer.'


def test_fitting_leaves_room_for_descenders_and_shadow_at_bottom():
    from meme_project import TextLayer,render_layers
    image=Image.new('RGBA',(1920,1080))
    layer=TextLayer(text='My therapist says I\'m stuck in the past until I hear this guitar solo.',
                    x=.04,y=.759,width=.58,size=24,font='DejaVuSerif.ttf',shadow=True)
    fitted=ai.fit_caption_layers(image,[layer])
    bounds=render_layers(image,fitted).getbbox()
    assert bounds and bounds[3]<image.height
    assert fitted[0].size>=16
    assert fitted[0].text==layer.text
