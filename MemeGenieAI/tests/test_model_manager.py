import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from model_manager import (
    ModelInfo, models, resolve_model, hash_file, verify_model, download_model,
    _safe_extract_zip, create_offline_bundle,
)


def info_for(payload: bytes) -> ModelInfo:
    return ModelInfo(
        id="mock", display_name="Mock Checkpoint", filename="sample.gguf",
        url="https://huggingface.co/example/sample/resolve/main/sample.gguf",
        sha256=hashlib.sha256(payload).hexdigest(), minimum_size_bytes=1,
        license_name="CreativeML OpenRAIL-M", license_url="https://huggingface.co/licence",
        model_card="https://huggingface.co/example", notes="Test-only", size_bytes=len(payload),
    )


class MockResponse:
    def __init__(self, content: bytes, status: int, headers=None):
        self.content = content
        self.status_code = status
        self.headers = headers or {"Content-Length": str(len(content))}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("Bad response")

    def iter_content(self, chunk_size):
        return [self.content[:2], self.content[2:]]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


class MockSession:
    def __init__(self, response):
        self.response = response
        self.headers = None

    def get(self, url, *, headers=None, **kwargs):
        self.headers = headers
        return self.response


def test_catalog_has_verified_real_references():
    found = models()
    assert len(found) >= 2
    assert found[0].id == "sd15-q4"
    assert resolve_model(found[0].id).filename.endswith(".gguf")
    assert all(len(model.sha256) == 64 for model in found)
    assert all(model.url.startswith("https://huggingface.co/") for model in found)


def test_model_download_verifies_and_atomic_rename(tmp_path):
    data = b"Genuine actual fake test bytes"
    info = info_for(data)
    session = MockSession(MockResponse(data, 200))
    result = download_model(info, tmp_path, session=session)
    assert result.read_bytes() == data
    assert verify_model(info, tmp_path)
    assert not result.with_name(result.name + ".part").exists()


def test_download_resumes_using_range(tmp_path):
    data = b"abcdef123456"
    info = info_for(data)
    partial = tmp_path / "sample.gguf.part"
    partial.write_bytes(data[:5])
    session = MockSession(MockResponse(data[5:], 206, {
        "Content-Length": str(len(data) - 5),
        "Content-Range": f"bytes 5-{len(data)-1}/{len(data)}"
    }))
    assert download_model(info, tmp_path, session=session).read_bytes() == data
    assert session.headers["Range"] == "bytes=5-"


def test_download_restarts_when_server_ignores_range(tmp_path):
    data = b"123456789ab"
    info = info_for(data)
    (tmp_path / "sample.gguf.part").write_bytes(b"old partial")
    session = MockSession(MockResponse(data, 200))
    assert download_model(info, tmp_path, session=session).read_bytes() == data


def test_download_corrupted_data_never_appears_installed(tmp_path):
    data = b"real checkpoint"
    info = info_for(data)
    session = MockSession(MockResponse(b"wrong model!!", 200))
    with pytest.raises(ValueError):
        download_model(info, tmp_path, session=session)
    assert not info.installed_path(tmp_path).exists()
    assert (tmp_path / "sample.gguf.part").is_file()


def test_cancel_preserves_partial(tmp_path):
    data = b"model bytes"
    info = info_for(data)
    checks = [False, False, True]

    def cancel():
        return checks.pop(0)

    with pytest.raises(InterruptedError):
        download_model(info, tmp_path,
                       session=MockSession(MockResponse(data, 200)), should_cancel=cancel)
    assert (tmp_path / "sample.gguf.part").exists()


def test_traversal_archive_rejected(tmp_path):
    archive = tmp_path / "attacker.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../../bad.txt", "bad")
    with pytest.raises(ValueError):
        _safe_extract_zip(archive, tmp_path / "extract")
    assert not (tmp_path / "bad.txt").exists()


def test_offline_bundle_rejects_missing_weight(tmp_path):
    with pytest.raises(FileNotFoundError):
        create_offline_bundle(tmp_path / "offline.zip", selected_ids=("sd15-q4",),
                              store=tmp_path / "models", engine_store=tmp_path / "engine")
    assert not (tmp_path / "offline.zip").exists()


def test_offline_bundle_contains_real_model_bytes_and_license(tmp_path, monkeypatch):
    import model_manager
    data = b"verified checkpoint contents"
    info = info_for(data)
    storage = tmp_path / "models"
    storage.mkdir()
    (storage / info.filename).write_bytes(data)
    (storage / "CreativeML-OpenRAIL-M-LICENSE.md").write_text("CreativeML Open RAIL-M license text")
    engine_storage = tmp_path / "engine"
    engine_storage.mkdir()
    engine_file = engine_storage / ("sd-cli.exe" if __import__('sys').platform == 'win32' else "sd-cli")
    engine_file.write_bytes(b"dummy engine test binary")
    monkeypatch.setattr(model_manager, "resolve_model", lambda name: info)
    archive = model_manager.create_offline_bundle(tmp_path / "complete-offline.zip",
                     selected_ids=("mock",), store=storage, engine_store=engine_storage)
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.read("MemeGenieAI/models/sample.gguf") == data
        assert zipped.read("MemeGenieAI/models/CreativeML-OpenRAIL-M-LICENSE.md")
        assert zipped.read("MemeGenieAI/engine/" + engine_file.name) == b"dummy engine test binary"
        assert "MemeGenieAI/memegenie_ai.py" in zipped.namelist()
