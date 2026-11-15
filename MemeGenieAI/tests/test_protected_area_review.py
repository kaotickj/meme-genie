import copy
from unittest.mock import patch
import pytest
from PIL import Image
from PyQt6.QtCore import QSettings,QPointF,Qt
from PyQt6.QtWidgets import QApplication,QDialog,QCheckBox,QPushButton,QDialogButtonBox
from PyQt6.QtTest import QTest
from memegenie_ai import MemeGeneratorApp,ProtectedAreaView
from meme_project import TextLayer,render_layers


def drag(view,start,end):
    bounds=view.scene().sceneRect()
    a=view.mapFromScene(QPointF(start[0]*bounds.width(),start[1]*bounds.height()))
    b=view.mapFromScene(QPointF(end[0]*bounds.width(),end[1]*bounds.height()))
    QTest.mousePress(view.viewport(),Qt.MouseButton.LeftButton,pos=a)
    QTest.mouseMove(view.viewport(),b)
    QTest.mouseRelease(view.viewport(),Qt.MouseButton.LeftButton,pos=b)


@pytest.mark.parametrize('size',[(1080,1920),(1920,1080)])
def test_drag_maps_to_image_not_letterboxed_viewport(size):
    app=QApplication.instance() or QApplication([])
    view=ProtectedAreaView();view.resize(850,600);view.show();view.show_pil(Image.new('RGB',size));app.processEvents()
    areas=[];view.area_selected.connect(areas.append);view.set_selecting(True)
    drag(view,(.8,.7),(.2,.3))
    assert len(areas)==1
    for key,value in [('x',.2),('y',.3),('width',.6),('height',.4)]:assert areas[0][key]==pytest.approx(value,abs=.007)
    view.close()


@pytest.mark.parametrize('accept',[False,True])
def test_review_protection_moves_copy_without_mutating_manual_or_model_layers(tmp_path,accept):
    app=QApplication.instance() or QApplication([])
    source=[dict(name='Idea',layers=[TextLayer(text='My solo has a copay.',x=.1,y=.5,width=.8,size=24)],subject_checked=True,protected_regions=[])]
    before=copy.deepcopy(source)
    with patch('memegenie_ai.QSettings',return_value=QSettings(str(tmp_path/'review.ini'),QSettings.Format.IniFormat)):
        window=MemeGeneratorApp();window.base_image=Image.new('RGBA',(1080,1920),'#444444');window._layout_background=window.base_image.copy()
        window.text_input.setPlainText('My manual caption');manual=copy.deepcopy(window.text_layers)
        def interact(dialog):
            dialog.show();app.processEvents()
            view=dialog.findChild(ProtectedAreaView);protect=next(c for c in dialog.findChildren(QCheckBox) if c.text()=='Keep an area clear')
            buttons=dialog.findChild(QDialogButtonBox);apply=buttons.button(QDialogButtonBox.StandardButton.Apply)
            protect.setChecked(True);assert not apply.isEnabled()
            drag(view,(.01,.35),(.99,.8))
            assert apply.isEnabled() and not protect.isChecked()
            dialog.hide()
            return QDialog.DialogCode.Accepted if accept else QDialog.DialogCode.Rejected
        with patch.object(QDialog,'exec',interact):window.preview_ai_layout('A musician',source)
        assert source==before
        if accept:
            mask=render_layers(Image.new('RGBA',window.base_image.size),window.text_layers)
            assert mask.crop((11,672,1069,1536)).getbbox() is None
            assert window.text_layers[0].text=='My solo has a copay.'
        else:assert window.text_layers==manual
        window.close()


def test_no_space_disables_apply_and_undo_restores_original(tmp_path):
    app=QApplication.instance() or QApplication([])
    with patch('memegenie_ai.QSettings',return_value=QSettings(str(tmp_path/'full.ini'),QSettings.Format.IniFormat)):
        window=MemeGeneratorApp();window.base_image=Image.new('RGBA',(800,450),'gray');window._layout_background=window.base_image.copy()
        layer=TextLayer(text='Keep the subject clear',size=24)
        def interact(dialog):
            dialog.show();app.processEvents();view=dialog.findChild(ProtectedAreaView)
            protect=next(c for c in dialog.findChildren(QCheckBox) if c.text()=='Keep an area clear')
            apply=dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Apply)
            protect.setChecked(True);drag(view,(.001,.001),(.999,.999))
            assert not apply.isEnabled()
            next(b for b in dialog.findChildren(QPushButton) if b.text()=='Undo area').click()
            assert apply.isEnabled();dialog.hide();return QDialog.DialogCode.Accepted
        with patch.object(QDialog,'exec',interact):window.preview_ai_layout('A subject',[dict(name='Idea',layers=[layer],subject_checked=True,protected_regions=[])])
        assert window.text_layers==[layer]
        window.close()
