"""Download, validate, and package real Stable Diffusion models.

No weights are synthesized, stubbed, or silently substituted. Network is used
only when the user requests an explicit install. Downloaded weights are verified
against public SHA-256 values recorded in model_catalog.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import requests

APP_NAME = "Meme Genie AI Studio"
BASE_DIR = Path(__file__).resolve().parent


def app_data_dir() -> Path:
    """Writable user data path. Bundled PyInstaller resources are read-only."""
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / "MemeGenieAI"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "MemeGenieAI"
    return Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))) / "memegenie-ai"


def model_dir() -> Path:
    return app_data_dir() / "models"


def engine_dir() -> Path:
    return app_data_dir() / "engine"


def catalog_path() -> Path:
    # Prefer a JSON shipped beside source/executable, then packaged PyInstaller resource.
    beside_executable = Path(sys.executable).resolve().parent / "model_catalog.json"
    beside_source = BASE_DIR / "model_catalog.json"
    resource = Path(getattr(sys, "_MEIPASS", str(BASE_DIR))) / "model_catalog.json"
    for path in (beside_source, beside_executable, resource):
        if path.is_file():
            return path
    raise FileNotFoundError("Model catalog missing. Reinstall Meme Genie AI Studio.")


@dataclass(frozen=True)
class ModelInfo:
    id: str
    display_name: str
    filename: str
    url: str
    sha256: str
    minimum_size_bytes: int
    license_name: str
    license_url: str
    model_card: str
    notes: str
    size_bytes: int | None = None

    def installed_path(self, root: Path | None = None) -> Path:
        return (root or model_dir()) / self.filename


def models() -> list[ModelInfo]:
    contents = json.loads(catalog_path().read_text(encoding="utf-8"))
    return [ModelInfo(**entry) for entry in contents["models"]]


def resolve_model(model_id: str) -> ModelInfo:
    for info in models():
        if info.id == model_id:
            return info
    raise ValueError(f"Unknown model: {model_id}")


def hash_file(path: Path, progress: Callable[[int, int | None, str], None] | None = None,
              should_cancel: Callable[[], bool] | None = None) -> str:
    total = path.stat().st_size
    read = 0
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            if should_cancel and should_cancel():
                raise InterruptedError("Verification canceled")
            hasher.update(chunk)
            read += len(chunk)
            if progress:
                progress(read, total, "Verifying SHA-256")
    return hasher.hexdigest()


def verify_model(info: ModelInfo, folder: Path | None = None,
                 progress: Callable[[int, int | None, str], None] | None = None,
                 should_cancel: Callable[[], bool] | None = None) -> bool:
    path = info.installed_path(folder)
    if not path.is_file() or path.stat().st_size < info.minimum_size_bytes:
        return False
    if info.size_bytes is not None and path.stat().st_size != info.size_bytes:
        return False
    return hash_file(path, progress, should_cancel) == info.sha256


def download_model(info: ModelInfo, folder: Path | None = None,
                   progress: Callable[[int, int | None, str], None] | None = None,
                   should_cancel: Callable[[], bool] | None = None,
                   session: requests.Session | None = None) -> Path:
    """Resume with HTTP Range when supported; verify SHA-256 before atomic rename.

    Downloads use a bounded memory buffer and .part files. A canceled transfer can
    be resumed. A server ignoring Range forces a safe restart instead of corruption.
    """
    folder = folder or model_dir()
    folder.mkdir(parents=True, exist_ok=True)
    dest = info.installed_path(folder)
    partial = folder / (info.filename + ".part")
    if dest.is_file() and verify_model(info, folder, progress, should_cancel):
        return dest
    if dest.is_file():
        dest.unlink()
    http = session or requests.Session()
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    if should_cancel and should_cancel():
        raise InterruptedError("Download canceled")
    # No arbitrary URL input in catalog. HTTPS host is fixed to huggingface.co.
    url = urlparse(info.url)
    if url.scheme != "https" or url.hostname != "huggingface.co":
        raise ValueError("Catalog URL must be an official HTTPS Hugging Face URL")
    with http.get(info.url, headers=headers, stream=True, timeout=(15, 90),
                  allow_redirects=True) as response:
        if response.status_code == 416:
            # A complete .part file can legitimately receive 416; verify it first.
            if partial.exists() and hash_file(partial, progress, should_cancel) == info.sha256:
                os.replace(partial, dest)
                return dest
            raise RuntimeError("Server refused resume; delete the .part file and retry")
        response.raise_for_status()
        if existing and response.status_code == 206:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {existing}-"):
                raise RuntimeError("Unexpected Content-Range, refusing to append")
            mode, current = "ab", existing
        elif response.status_code == 200:
            mode, current = "wb", 0
        else:
            raise RuntimeError(f"Unexpected HTTP status {response.status_code}")
        length = response.headers.get("Content-Length")
        expected = current + int(length) if length and length.isdecimal() else None
        with partial.open(mode) as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if should_cancel and should_cancel():
                    raise InterruptedError("Download canceled. Progress kept for resume.")
                if not chunk:
                    continue
                output.write(chunk)
                current += len(chunk)
                if progress:
                    progress(current, expected, "Downloading model")
    if partial.stat().st_size < info.minimum_size_bytes:
        raise ValueError("Model download is incomplete; .part retained for retry")
    if info.size_bytes and partial.stat().st_size != info.size_bytes:
        raise ValueError("Downloaded file size does not match published model size")
    if hash_file(partial, progress, should_cancel) != info.sha256:
        raise ValueError("SHA-256 mismatch! Model not installed. Remove .part before retrying.")
    os.replace(partial, dest)
    if progress:
        progress(dest.stat().st_size, dest.stat().st_size, "Model installed and verified")
    # Licence text is fetched from the published model repository in a separate
    # setup action, and is required for offline redistribution.
    return dest


def retrieve_model_license(store: Path | None = None,
                           session: requests.Session | None = None) -> Path:
    """Fetch CreativeML OpenRAIL-M license text for redistribution."""
    store = store or model_dir()
    store.mkdir(parents=True, exist_ok=True)
    target = store / "CreativeML-OpenRAIL-M-LICENSE.md"
    if target.exists() and "CreativeML Open RAIL-M" in target.read_text(encoding="utf-8"):
        return target
    url = "https://huggingface.co/kostakoff/stable-diffusion-v1-5-GGUF/resolve/main/LICENSE.md"
    result = (session or requests.Session()).get(url, timeout=(15, 45))
    result.raise_for_status()
    if "CreativeML Open RAIL-M" not in result.text or len(result.content) < 8000:
        raise ValueError("Source did not provide a complete CreativeML OpenRAIL-M license")
    partial = target.with_suffix(".part")
    partial.write_bytes(result.content)
    os.replace(partial, target)
    return target


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    dest_real = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for info in zipped.infolist():
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(dest_real):
                raise ValueError("Unsafe archive: path outside engine directory")
            # Symlinks are not needed for Windows engine archives.
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe archive: symlink entry")
            if not info.is_dir() and (info.file_size > 600 * 1024 * 1024):
                raise ValueError("Unexpectedly large engine archive file")
        zipped.extractall(destination)


def _engine_files(folder: Path) -> list[Path]:
    name = "sd-cli.exe" if sys.platform == "win32" else "sd-cli"
    return sorted(folder.rglob(name), key=lambda path: len(path.parts)) if folder.exists() else []


def locate_engine(folder: Path | None = None) -> Path | None:
    for candidate in _engine_files(folder or engine_dir()):
        if candidate.is_file():
            return candidate
    return None


def get_github_engine_asset(flavor: str = "vulkan", session: requests.Session | None = None) -> dict:
    """Resolve release *asset* via official GitHub API; don't guess release URLs."""
    if flavor not in {"vulkan", "cpu"}:
        raise ValueError("Engine flavor must be vulkan or cpu")
    if sys.platform != "win32":
        raise RuntimeError("Automatic engine installation currently targets Windows x64. Linux: build with -DSD_VULKAN=ON or choose an installed executable.")
    http = session or requests.Session()
    req = http.get("https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/latest",
                   timeout=20, headers={"Accept": "application/vnd.github+json"})
    req.raise_for_status()
    data = req.json()
    suffix = f"-bin-win-{flavor}-x64.zip"
    matched = [a for a in data.get("assets", []) if a.get("name", "").endswith(suffix)]
    if not matched:
        raise RuntimeError(f"No Windows {flavor} x64 asset found on the current GitHub release")
    return matched[0]


def install_engine(flavor: str = "vulkan", folder: Path | None = None,
                   progress: Callable[[int, int | None, str], None] | None = None,
                   should_cancel: Callable[[], bool] | None = None,
                   session: requests.Session | None = None) -> Path:
    """Obtain official Windows Vulkan/CPU release ZIP and unpack locally."""
    folder = folder or engine_dir()
    folder.mkdir(parents=True, exist_ok=True)
    http = session or requests.Session()
    asset = get_github_engine_asset(flavor, http)
    url = asset["browser_download_url"]
    parts = urlparse(url)
    if parts.scheme != "https" or parts.hostname != "github.com":
        raise ValueError("Unexpected engine release download host")
    archive = folder / asset["name"]
    temp = folder / (asset["name"] + ".part")
    bytes_so_far = 0
    with http.get(url, stream=True, timeout=(15, 90)) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0)) or None
        with temp.open("wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                if should_cancel and should_cancel():
                    raise InterruptedError("Engine download canceled")
                if not chunk:
                    continue
                output.write(chunk)
                bytes_so_far += len(chunk)
                if progress:
                    progress(bytes_so_far, total, "Downloading official engine")
    digest = asset.get("digest", "")
    if digest.startswith("sha256:") and hash_file(temp, progress, should_cancel) != digest[7:]:
        temp.unlink(missing_ok=True)
        raise ValueError("Engine archive SHA-256 mismatch")
    os.replace(temp, archive)
    dest = folder / flavor
    # Do not extract into previously installed files or outside the destination.
    staging = folder / (flavor + "-staging")
    if staging.exists():
        shutil.rmtree(staging)
    _safe_extract_zip(archive, staging)
    found = _engine_files(staging)
    if not found:
        raise RuntimeError("Engine archive doesn't contain sd-cli.exe; please choose a different release")
    if dest.exists():
        shutil.rmtree(dest)
    staging.rename(dest)
    archive.unlink(missing_ok=True)
    return _engine_files(dest)[0]


def create_offline_bundle(target_zip: Path, *, require_engine: bool = True,
                          selected_ids: tuple[str, ...] = ("sd15-q4",),
                          store: Path | None = None, engine_store: Path | None = None,
                          progress: Callable[[int, int | None, str], None] | None = None) -> Path:
    """Package actual VERIFIED model bytes + local engine alongside source code.

    This deliberately will not create a misleading 'offline' archive with missing
    model weights. Caller must have downloaded model and optional engine first.
    """
    store = store or model_dir()
    engine_store = engine_store or engine_dir()
    verified: list[tuple[ModelInfo, Path]] = []
    for model_id in selected_ids:
        info = resolve_model(model_id)
        if not verify_model(info, store, progress):
            raise FileNotFoundError(f"Required model not installed/verified: {info.display_name}")
        verified.append((info, info.installed_path(store)))
    engine = locate_engine(engine_store)
    if require_engine and engine is None:
        raise FileNotFoundError("Engine must be installed to build a fully offline bundle")
    target_zip.parent.mkdir(parents=True, exist_ok=True)
    files = ["memegenie_ai.py", "sd_backend.py", "model_manager.py", "model_catalog.json",
             "setup_models.py", "diagnose_engine.py", "requirements.txt", "README.md", "run_windows.bat", "run_diagnostics_windows.bat", "run_linux.sh",
             "build_windows.bat", "build_linux.sh", "build_macos.sh", "run_macos.sh",
             "build.py", "MemeGenieAI.spec", "requirements-dev.txt", "docs/ENGINE_LICENSE.txt"]
    source_dir = BASE_DIR
    if getattr(sys, "frozen", False):
        source_dir = Path(getattr(sys, "_MEIPASS", str(BASE_DIR))) / "source"
    if any(not (source_dir / file).is_file() for file in files):
        raise FileNotFoundError("Source and license files required for offline bundle are missing. "
                                "Run the full source distribution or use setup_models.py with Python.")
    license_file = store / "CreativeML-OpenRAIL-M-LICENSE.md"
    if not license_file.is_file():
        raise FileNotFoundError("Model license missing; run setup_models.py license before redistributing")
    with zipfile.ZipFile(target_zip, "w", allowZip64=True, compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1) as zipped:
        for filename in files:
            zipped.write(source_dir / filename, "MemeGenieAI/" + filename)
        if getattr(sys, "frozen", False):
            app_executable = Path(sys.executable).resolve()
            zipped.write(app_executable, "MemeGenieAI/" + app_executable.name)
            internal = source_dir / "_internal"
            if internal.is_dir():
                for child in internal.rglob("*"):
                    if child.is_file():
                        zipped.write(child, "MemeGenieAI/_internal/" + child.relative_to(internal).as_posix())
        zipped.write(license_file, "MemeGenieAI/models/CreativeML-OpenRAIL-M-LICENSE.md")
        for info, path in verified:
            # Compressed model files are effectively incompressible; ZIP_STORED is faster.
            zipped.write(path, "MemeGenieAI/models/" + info.filename, compress_type=zipfile.ZIP_STORED)
        if engine:
            # Keep DLLs, shaders and support files alongside sd-cli.
            base = engine.parent
            for candidate in base.rglob("*"):
                if candidate.is_file():
                    zipped.write(candidate, "MemeGenieAI/engine/" + candidate.relative_to(base).as_posix())
    return target_zip
