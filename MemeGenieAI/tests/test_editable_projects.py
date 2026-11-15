from dataclasses import asdict, replace
import json
import zipfile
from unittest.mock import patch
import pytest
from PIL import Image, ImageChops
from meme_project import TextLayer, save_project, load_project, render_layers


def test_portable_roundtrip_and_source_preserved(tmp_path):
    background = Image.new('RGBA', (512, 512), '#334455')
    layers = [TextLayer(text='TOP', y=.05), TextLayer(text='BOTTOM', y=.8, color=(255,200,0,255))]
    pixels = background.tobytes()
    target = tmp_path/'sample.mgproject'
    save_project(target, background, layers, True)
    reopened, restored, credit = load_project(target)
    assert reopened.tobytes() == pixels and restored == layers and credit
    assert render_layers(reopened, restored).tobytes() == render_layers(background, layers).tobytes()
    assert background.tobytes() == pixels
    assert ImageChops.difference(background.convert('RGB'), render_layers(background,layers).convert('RGB')).getbbox()


@pytest.mark.parametrize('update', [{'x':float('nan')}, {'x':.95,'width':.9}, {'font':'/etc/passwd'}, {'size':999}, {'color':[999,0,0,0]}, {'shadow':'yes'}])
def test_invalid_layer_rejected(update):
    with pytest.raises(ValueError): TextLayer.parse({**asdict(TextLayer()), **update})


def test_invalid_project_and_interrupted_save(tmp_path):
    target=tmp_path/'a.mgproject'; background=Image.new('RGB',(20,20))
    save_project(target,background,[TextLayer()]); before=target.read_bytes()
    with patch('meme_project.os.replace',side_effect=OSError('test')):
        with pytest.raises(OSError): save_project(target,background,[TextLayer(text='new')])
    assert target.read_bytes()==before
    assert not list(tmp_path.glob('.meme-project-*'))
    bad=tmp_path/'bad.mgproject'
    with zipfile.ZipFile(bad,'w') as z: z.writestr('../outside','bad')
    with pytest.raises(ValueError):load_project(bad)


def test_gui_layers_save_open_cancel_and_export(tmp_path):
    from PyQt6.QtWidgets import QApplication,QFileDialog,QMessageBox
    from PyQt6.QtCore import QSettings
    from memegenie_ai import MemeGeneratorApp
    app=QApplication.instance() or QApplication([])
    settings=QSettings(str(tmp_path/'settings.ini'),QSettings.Format.IniFormat)
    background=tmp_path/'base.png';Image.new('RGB',(512,512),'#334455').save(background)
    with patch('memegenie_ai.QSettings',return_value=settings):
        w=MemeGeneratorApp();w.load_base_image(str(background),False)
        w.auto_layout_checkbox.setChecked(False)
        w.text_input.setPlainText('First');w.layer_y.setValue(5)
        w.add_text_layer();w.text_input.setPlainText('Second');w.layer_y.setValue(75)
        w.layer_choice.setCurrentIndex(0)
        assert w.text_input.toPlainText()=='First' and w.layer_y.value()==5
        expected=w.render_meme().tobytes();target=tmp_path/'two.mgproject'
        with patch.object(QFileDialog,'getSaveFileName',return_value=(str(target),'Meme')):w.save_editable_project()
        w.text_input.setPlainText('Unsaved')
        with patch.object(QFileDialog,'getOpenFileName',return_value=(str(target),'Meme')),patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.No):w.open_editable_project()
        assert w.text_input.toPlainText()=='Unsaved'
        with patch.object(QFileDialog,'getOpenFileName',return_value=(str(target),'Meme')),patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.Yes):w.open_editable_project()
        assert len(w.text_layers)==2 and w.render_meme().tobytes()==expected
        output=tmp_path/'flat.png'
        with patch.object(QFileDialog,'getSaveFileName',return_value=(str(output),'PNG')),patch.object(QMessageBox,'information'):w.generate_meme()
        with Image.open(output) as im: assert im.convert('RGBA').tobytes()==expected
        from PIL import ImageOps
        wide=tmp_path/'wide.png'
        with patch.object(QFileDialog,'getSaveFileName',return_value=(str(wide),'PNG')):
            w.export_sized(1920,1080)
        fitted=ImageOps.fit(w.base_image,(1920,1080),method=Image.Resampling.LANCZOS,centering=(.5,.5))
        with Image.open(wide) as im:
            assert im.size==(1920,1080)
            assert im.convert('RGB').tobytes()==w.render_meme(fitted).convert('RGB').tobytes()
        with patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.Yes):w.remove_text_layer()
        assert len(w.text_layers)==1
        w.close()


def test_mandatory_original_credit_all_exports(tmp_path):
    from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox
    from PyQt6.QtCore import QSettings
    from PIL import ImageDraw
    from memegenie_ai import MemeGeneratorApp
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path/'credit.ini'), QSettings.Format.IniFormat)
    background = tmp_path/'source.png'
    Image.new('RGB', (512,512), '#334455').save(background)
    with patch('memegenie_ai.QSettings', return_value=settings):
        w = MemeGeneratorApp(); w.load_base_image(str(background), False)
        assert not hasattr(w, 'attribution_checkbox')
        before = w.base_image.tobytes()
        original_text = ImageDraw.ImageDraw.text
        recorded = []
        def capture(self, xy, text, *args, **kwargs):
            recorded.append(text)
            return original_text(self, xy, text, *args, **kwargs)
        with patch.object(ImageDraw.ImageDraw, 'text', capture):
            w.render_meme()
        assert 'https://github.com/kaotickj' in ' '.join(recorded)
        expected = w.render_meme().convert('RGB')
        for ext in ('png','jpg'):
            target = tmp_path/f'export.{ext}'
            with patch.object(QFileDialog,'getSaveFileName',return_value=(str(target),'PNG')), patch.object(QMessageBox,'information'):
                w.generate_meme()
            with Image.open(target) as im:
                assert im.size == (512,512)
                assert ImageChops.difference(im.convert('RGB').crop((0,460,512,512)), Image.open(background).crop((0,460,512,512))).getbbox()
        base = tmp_path/'credited-base.png'
        with patch.object(QFileDialog,'getSaveFileName',return_value=(str(base),'PNG')):
            w.save_base_image()
        with Image.open(base) as im: assert im.convert('RGB').tobytes() == expected.tobytes()
        assert w.base_image.tobytes() == before
        # A legacy false preference must not disable required attribution.
        project = tmp_path/'legacy.mgproject'
        save_project(project,w.base_image,w.text_layers,False)
        with zipfile.ZipFile(project) as z: assert json.loads(z.read('project.json'))['attribution'] is True
        w.close()


def test_simple_workflow_auto_layout_no_model(tmp_path):
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QSettings
    from memegenie_ai import MemeGeneratorApp
    app=QApplication.instance() or QApplication([])
    with patch('memegenie_ai.QSettings', return_value=QSettings(str(tmp_path/'simple.ini'),QSettings.Format.IniFormat)):
        w=MemeGeneratorApp()
        assert not w.tabs.isTabVisible(1)
        assert not w.text_details.isHidden()
        assert not w.auto_layout_checkbox.isChecked()
        w.auto_layout_checkbox.setChecked(True)
        w.base_image=Image.new('RGBA',(512,512),'#777777')
        w.text_input.setPlainText('ONE GOOD IDEA')
        assert w.layer_y.value()==5 and w.layer_width.value()==90
        assert w.ai_process is None
        w.add_text_layer();w.text_input.setPlainText('Now make it a meme')
        assert w.text_layers[0].y < w.text_layers[1].y
        assert w.render_meme() is not None
        w.show();app.processEvents()
        w.grab().save(str(tmp_path/'simple-ui.png'))
        w.close()


def test_review_regressions_and_library_clear(tmp_path):
    from PyQt6.QtWidgets import QApplication,QMessageBox
    from PyQt6.QtCore import QSettings
    from memegenie_ai import MemeGeneratorApp
    from meme_project import validate_image
    app=QApplication.instance() or QApplication([])
    with patch('memegenie_ai.QSettings',return_value=QSettings(str(tmp_path/'review.ini'),QSettings.Format.IniFormat)):
        w=MemeGeneratorApp();w.base_image=Image.new('RGBA',(512,512),'gray')
        layer=TextLayer(text='THIS CAPTION STAYS',y=.95,size=16,font='missing-font.ttf',x=.0545,width=.8955)
        w.text_layers=[layer];w.refresh_layer_choices()
        assert w.text_layers[0]==layer
        full=w.render_meme()
        blank=w.add_attribution(w.base_image)
        assert full.tobytes()!=blank.tobytes()
        folder=tmp_path/'pictures';folder.mkdir()
        for name in ('original.png','generated.jpg'):Image.new('RGB',(32,32),'red').save(folder/name)
        w.output_dir.setText(str(folder));w.refresh_gallery();assert w.gallery.count()==2
        before=w.base_image.tobytes();checks={p.name:p.read_bytes() for p in folder.iterdir()}
        with patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.No):assert w.clear_gallery()==0
        assert w.clear_gallery(confirm=False)==2 and w.gallery.count()==0
        assert {p.name:p.read_bytes() for p in folder.iterdir()}==checks and w.base_image.tobytes()==before
        Image.new('RGB',(32,32),'blue').save(folder/'new.png');w.refresh_gallery();assert w.gallery.count()==1
        w.restore_gallery();assert w.gallery.count()==3
        w.close()
    fake=type('ImageShape',(),{'size':(1,25_000_000),'width':1,'height':25_000_000})()
    with pytest.raises(ValueError):validate_image(fake)


def test_controls_can_revert_to_original_values(tmp_path):
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QSettings
    from memegenie_ai import MemeGeneratorApp
    app=QApplication.instance() or QApplication([])
    with patch('memegenie_ai.QSettings',return_value=QSettings(str(tmp_path/'revert.ini'),QSettings.Format.IniFormat)):
        w=MemeGeneratorApp();w.base_image=Image.new('RGBA',(512,512),'gray')
        w.text_layers=[TextLayer(text='Original',x=.05,size=32)];w.refresh_layer_choices()
        w.text_input.setPlainText('Changed');w.layer_x.setValue(6);w.font_size.setValue(40)
        w.text_input.setPlainText('Original');w.layer_x.setValue(5);w.font_size.setValue(32)
        assert w.text_layers[0].text=='Original' and w.text_layers[0].x==.05 and w.text_layers[0].size==32
        w.close()


@pytest.mark.parametrize('index,dimensions',[(1,(1920,1080)),(2,(1080,1920))])
def test_orientation_preview_exactly_matches_exports(tmp_path,index,dimensions):
    from PyQt6.QtWidgets import QApplication,QFileDialog,QMessageBox
    from PyQt6.QtCore import QSettings
    from memegenie_ai import MemeGeneratorApp
    app=QApplication.instance() or QApplication([])
    with patch('memegenie_ai.QSettings',return_value=QSettings(str(tmp_path/'orientation.ini'),QSettings.Format.IniFormat)):
        w=MemeGeneratorApp();w.base_image=Image.new('RGBA',(800,600),'#456789')
        w.text_input.setPlainText('Preview matches the export')
        original=w.base_image.tobytes()
        w.preview_format.setCurrentIndex(index)
        expected=w.selected_preview_image()
        assert expected.size==dimensions
        regular=tmp_path/'selected.png'
        with patch.object(QFileDialog,'getSaveFileName',return_value=(str(regular),'PNG')),patch.object(QMessageBox,'information'):w.generate_meme()
        fixed=tmp_path/'preset.png'
        with patch.object(QFileDialog,'getSaveFileName',return_value=(str(fixed),'PNG')):w.export_sized(*dimensions)
        for path in (regular,fixed):
            with Image.open(path) as im:assert im.convert('RGB').tobytes()==expected.convert('RGB').tobytes()
        assert w.base_image.tobytes()==original and w.base_image.size==(800,600)
        w.preview_format.setCurrentIndex(0);assert w.selected_preview_image().size==(800,600)
        w.close()
