from pathlib import Path
import model_manager


def test_macos_data_path(monkeypatch):
    monkeypatch.setattr(model_manager.sys, 'platform', 'darwin')
    assert model_manager.app_data_dir() == Path.home() / 'Library' / 'Application Support' / 'MemeGenieAI'


def test_linux_data_override(monkeypatch, tmp_path):
    monkeypatch.setattr(model_manager.sys, 'platform', 'linux')
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path))
    assert model_manager.app_data_dir() == tmp_path / 'memegenie-ai'
