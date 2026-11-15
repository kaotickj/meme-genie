import copy
import json
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image, ImageDraw, ImageFont
from PyQt6.QtCore import QPointF, QSettings, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QPushButton
from ai_meme_layout import NoCaptionSpace, place_around_subjects, subject_clearance_regions, suggest_layout
from memegenie_ai import MemeGeneratorApp, ProtectedAreaView
from meme_project import TextLayer, render_layers, wrap_text_lines
from test_protected_area_review import drag


def check(dialog, label):
    return next(w for w in dialog.findChildren(QCheckBox) if w.text()==label)


def button(dialog, label):
    return next(w for w in dialog.findChildren(QPushButton) if w.text()==label)


def controls(dialog):
    return (dialog.findChild(ProtectedAreaView),
            dialog.findChild(QComboBox, 'protectedRegionChoice'),
            dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Apply))


@pytest.mark.parametrize('accept',[False,True])
def test_remove_false_detection_allows_compact_portrait_without_mutating_input(tmp_path,accept):
    app=QApplication.instance() or QApplication([])
    regions=[dict(label='False detection',x=.4,y=.02,width=.56,height=.8),
             dict(label='Drum kit',x=0,y=.55,width=1,height=.45,reviewed=True)]
    layer=TextLayer(text='My therapist said open up. The band said turn down.',x=.04,y=.04,width=.33,size=25)
    source=[dict(name=name,layers=[copy.deepcopy(layer)],protected_regions=copy.deepcopy(regions),
                 placement_error='Review the subject areas') for name in ['One','Two']]
    original=copy.deepcopy(source)
    with patch('memegenie_ai.QSettings',return_value=QSettings(str(tmp_path/'review.ini'),QSettings.Format.IniFormat)):
        window=MemeGeneratorApp();window.base_image=Image.new('RGBA',(1080,1920),'#444444');window._layout_background=window.base_image.copy()
        manual=copy.deepcopy(window.text_layers)
        def interact(dialog):
            dialog.show();app.processEvents();view,choices,apply=controls(dialog)
            assert check(dialog,'Review subject areas').isChecked() and not apply.isEnabled()
            assert choices.count()==2 and 'AI' in choices.itemText(0)
            button(dialog,'Remove area').click()
            assert apply.isEnabled() and choices.count()==1
            alternatives=next(w for w in dialog.findChildren(QComboBox) if w is not choices)
            alternatives.setCurrentIndex(1)
            assert choices.count()==1 and apply.isEnabled()
            button(dialog,'Undo area').click()
            assert choices.count()==2 and not apply.isEnabled()
            choices.setCurrentIndex(0);button(dialog,'Remove area').click()
            dialog.hide();return QDialog.DialogCode.Accepted if accept else QDialog.DialogCode.Rejected
        with patch.object(QDialog,'exec',interact):window.preview_ai_layout('A subject',source)
        assert source==original
        if accept:
            assert window.text_layers[0].text==layer.text and window.text_layers[0].width>=.65
            assert render_layers(Image.new('RGBA',window.base_image.size),window.text_layers).crop((0,1056,1080,1920)).getbbox() is None
        else:assert window.text_layers==manual
        window.close()


def test_redraw_head_uses_reviewed_bounds_and_undo_restores_model_box(tmp_path):
    app=QApplication.instance() or QApplication([])
    region=dict(label='Head',x=.1,y=.1,width=.6,height=.7)
    source=[dict(name='Idea',layers=[TextLayer(text='Keep these exact words')],protected_regions=[region],placement_error='No room')]
    original=copy.deepcopy(source)
    with patch('memegenie_ai.QSettings',return_value=QSettings(str(tmp_path/'redraw.ini'),QSettings.Format.IniFormat)):
        window=MemeGeneratorApp();window.base_image=Image.new('RGBA',(800,450),'gray');window._layout_background=window.base_image.copy()
        def interact(dialog):
            dialog.show();app.processEvents();view,choices,apply=controls(dialog)
            button(dialog,'Redraw area').click();assert not apply.isEnabled()
            drag(view,(.7,.5),(.95,.9))
            assert 'Reviewed: Head' in choices.currentText()
            assert view._areas[0]['reviewed'] and view._areas[0]['x']==pytest.approx(.7,abs=.007)
            box=subject_clearance_regions(view._areas)[0]
            assert box['y']+box['height']<.94
            assert apply.isEnabled()
            button(dialog,'Undo area').click()
            assert view._areas==[region]
            dialog.hide();return QDialog.DialogCode.Rejected
        with patch.object(QDialog,'exec',interact):window.preview_ai_layout('A subject',source)
        assert source==original;window.close()


@pytest.mark.parametrize('size',[(1080,1920),(1920,1080)])
def test_review_click_selects_smallest_box_at_actual_image_coordinates(size):
    app=QApplication.instance() or QApplication([])
    view=ProtectedAreaView();view.resize(850,600);view.show();view.show_pil(Image.new('RGB',size))
    areas=[dict(label='Large',x=.1,y=.1,width=.8,height=.8),dict(label='Small',x=.3,y=.3,width=.2,height=.2)]
    view.show_areas(areas);view.reviewing=True;app.processEvents();selected=[];view.area_clicked.connect(selected.append)
    p=view.mapFromScene(QPointF(size[0]*.4,size[1]*.4))
    QTest.mouseClick(view.viewport(),Qt.MouseButton.LeftButton,pos=p)
    assert selected==[1];view.close()


def test_redraw_cancellation_and_alternative_switch_do_not_edit_regions(tmp_path):
    app=QApplication.instance() or QApplication([])
    source=[dict(name=name,layers=[TextLayer(text='My exact caption')],subject_checked=True,
                 protected_regions=[dict(label='Subject',x=.7,y=.5,width=.2,height=.3)]) for name in ['One','Two']]
    with patch('memegenie_ai.QSettings',return_value=QSettings(str(tmp_path/'cancel.ini'),QSettings.Format.IniFormat)):
        window=MemeGeneratorApp();window.base_image=Image.new('RGBA',(800,450),'gray');window._layout_background=window.base_image.copy()
        def interact(dialog):
            dialog.show();app.processEvents();view,choices,apply=controls(dialog)
            check(dialog,'Review subject areas').setChecked(True)
            button(dialog,'Redraw area').click();assert not apply.isEnabled()
            check(dialog,'Keep an area clear').setChecked(False)
            assert apply.isEnabled() and not button(dialog,'Undo area').isEnabled()
            button(dialog,'Redraw area').click()
            alternatives=next(w for w in dialog.findChildren(QComboBox) if w is not choices)
            alternatives.setCurrentIndex(1)
            assert apply.isEnabled() and not check(dialog,'Keep an area clear').isChecked()
            assert view._areas==source[1]['protected_regions']
            dialog.hide();return QDialog.DialogCode.Rejected
        with patch.object(QDialog,'exec',interact):window.preview_ai_layout('A subject',source)
        window.close()


def test_full_image_detection_is_reviewable_but_cannot_apply_until_corrected(tmp_path):
    app=QApplication.instance() or QApplication([])
    record=dict(text='My exact caption',x=.05,y=.05,width=.9,size=24,align='center',font_style='bold',color='white')
    region=dict(label='Bad box',x=0,y=0,width=1,height=1)
    payload=dict(image_description='A subject',protected_regions=[region],suggestions=[dict(name=name,layers=[{**record,'font_style':style}]) for name,style in [('One','bold'),('Two','serif')]])
    session=MagicMock();session.__enter__.return_value=session
    session.get.return_value.json.return_value={'models':[{'name':'gemma3:4b','capabilities':['vision']}]}
    session.post.return_value.content=b'{}';session.post.return_value.json.return_value={'message':{'content':json.dumps(payload)}}
    canvas=Image.new('RGBA',(800,450),'gray')
    with patch('ai_meme_layout.requests.Session',return_value=session):
        _,options=suggest_layout(canvas,'Use exactly "My exact caption"',alternatives=True)
    assert all(o.get('placement_error') and not o.get('subject_checked') for o in options)
    with patch('memegenie_ai.QSettings',return_value=QSettings(str(tmp_path/'blocked.ini'),QSettings.Format.IniFormat)):
        window=MemeGeneratorApp();window.base_image=canvas;window._layout_background=canvas.copy()
        def interact(dialog):
            dialog.show();app.processEvents();view,choices,apply=controls(dialog)
            assert not apply.isEnabled()
            button(dialog,'Redraw area').click();drag(view,(.7,.5),(.95,.95))
            assert apply.isEnabled()
            button(dialog,'Undo area').click();assert not apply.isEnabled()
            # Even an artificial Accepted dialog cannot bypass the apply guard.
            dialog.hide();return QDialog.DialogCode.Accepted
        manual=copy.deepcopy(window.text_layers)
        with patch.object(QDialog,'exec',interact):window.preview_ai_layout('A subject',options)
        assert window.text_layers==manual;window.close()


def test_reviewed_head_does_not_receive_guessed_chin_extension():
    box=dict(label='Head',x=.2,y=.2,width=.2,height=.2)
    guessed=subject_clearance_regions([box])[0]
    reviewed=subject_clearance_regions([{**box,'reviewed':True}])[0]
    assert guessed['y']+guessed['height']==pytest.approx(.52)
    assert reviewed['y']+reviewed['height']==pytest.approx(.425)
    assert 'reviewed' not in box


def test_portrait_rejects_seven_line_column_instead_of_hiding_no_space():
    image=Image.new('RGB',(1080,1920),'gray')
    text='"Sorry I\'m late, the mountain needed a good beat." - My therapist.'
    layer=TextLayer(text=text,x=.04,y=.04,width=.33,size=25,font='DejaVuSerif-Bold.ttf')
    regions=[dict(label='Right',x=.4,y=0,width=.6,height=1),dict(label='Bottom',x=0,y=.55,width=1,height=.45)]
    with pytest.raises(NoCaptionSpace):place_around_subjects(image,[layer],regions)
    result=place_around_subjects(image,[layer],regions[1:])[0]
    font=ImageFont.truetype(result.font,round(result.size*image.width/512))
    lines=wrap_text_lines(ImageDraw.Draw(image),result.text,font,round(result.width*image.width))
    assert len(lines)<=4 and result.width>=.65 and result.text==text


def test_four_added_areas_allow_redraw_but_cancel_cannot_add_a_fifth(tmp_path):
    app=QApplication.instance() or QApplication([])
    source=[dict(name='Idea',layers=[TextLayer(text='Short caption')],subject_checked=True,protected_regions=[])]
    with patch('memegenie_ai.QSettings',return_value=QSettings(str(tmp_path/'limit.ini'),QSettings.Format.IniFormat)):
        window=MemeGeneratorApp();window.base_image=Image.new('RGBA',(800,450),'gray');window._layout_background=window.base_image.copy()
        def interact(dialog):
            dialog.show();app.processEvents();view,choices,apply=controls(dialog)
            check(dialog,'Review subject areas').setChecked(True)
            for x in (.1,.3,.5,.7):
                check(dialog,'Keep an area clear').setChecked(True);drag(view,(x,.7),(x+.1,.85))
            assert choices.count()==4 and not check(dialog,'Keep an area clear').isEnabled()
            button(dialog,'Redraw area').click()
            assert check(dialog,'Keep an area clear').isEnabled() and not apply.isEnabled()
            check(dialog,'Keep an area clear').setChecked(False)
            assert not check(dialog,'Keep an area clear').isEnabled()
            button(dialog,'Remove area').click()
            assert choices.count()==3 and check(dialog,'Keep an area clear').isEnabled()
            button(dialog,'Undo area').click()
            assert choices.count()==4 and not check(dialog,'Keep an area clear').isEnabled()
            dialog.hide();return QDialog.DialogCode.Rejected
        with patch.object(QDialog,'exec',interact):window.preview_ai_layout('A subject',source)
        window.close()
