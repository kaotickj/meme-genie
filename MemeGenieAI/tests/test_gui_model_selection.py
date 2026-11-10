import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from PyQt6.QtWidgets import QApplication, QComboBox, QLineEdit, QLabel
from memegenie_ai import MemeGeneratorApp


def test_catalog_switch_finds_browsed_model_sibling(tmp_path):
    app = QApplication.instance() or QApplication([])
    old = tmp_path / "old.gguf"
    new = tmp_path / "full.safetensors"
    old.write_bytes(b"test")
    new.write_bytes(b"test")
    info = SimpleNamespace(minimum_size_bytes=1, installed_path=lambda folder: folder / new.name,
                           license_name="test", license_url="test", display_name="Full")
    dummy = SimpleNamespace(model_choice=QComboBox(), sd_model=QLineEdit(str(old)),
                            install_status=QLabel(), model_license=QLabel())
    dummy.model_choice.addItem("Full", "full")
    with patch("memegenie_ai.resolve_model", return_value=info):
        MemeGeneratorApp.on_model_choice(dummy)
    assert dummy.sd_model.text() == str(new)
    assert "Found model" in dummy.install_status.text()


def test_instrument_preset_excludes_people():
    from PyQt6.QtWidgets import QTextEdit
    app = QApplication.instance() or QApplication([])
    dummy = SimpleNamespace(prompt_preset=QComboBox(), ai_prompt=QTextEdit(),
                            ai_negative=QTextEdit(), dim_preset=QComboBox())
    dummy.prompt_preset.addItem("Instrument", "instrument")
    dummy.dim_preset.addItem("512", (512, 512))
    MemeGeneratorApp.apply_prompt_preset(dummy)
    assert "one upright double bass" in dummy.ai_prompt.toPlainText()
    assert "people" in dummy.ai_negative.toPlainText()
    assert dummy.dim_preset.currentIndex() == 0
