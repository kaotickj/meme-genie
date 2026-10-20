# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from pathlib import Path

root = Path(SPECPATH)
source_files = ['memegenie_ai.py', 'sd_backend.py', 'model_manager.py',
                'model_catalog.json', 'setup_models.py', 'diagnose_engine.py',
                'requirements.txt', 'requirements-dev.txt', 'README.md',
                'run_windows.bat', 'run_diagnostics_windows.bat', 'run_linux.sh',
                'run_macos.sh', 'build_windows.bat', 'build_linux.sh',
                'build_macos.sh', 'build.py', 'MemeGenieAI.spec', 'docs/ENGINE_LICENSE.txt']
a = Analysis(
    ['memegenie_ai.py'],
    pathex=[],
    binaries=[],
    datas=[(str(root / 'model_catalog.json'), '.'),
           (str(root / 'docs' / 'ENGINE_LICENSE.txt'), 'docs')] +
          [(str(root / f), str(Path('source') / Path(f).parent)) for f in source_files],
    hiddenimports=['model_manager'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
# Qt 6.11 on Windows uses the OS ICU API. A same-named DLL from PATH
# (e.g. MSYS2) has versioned exports and breaks QtCore in frozen builds.
if sys.platform == 'win32':
    a.binaries = [entry for entry in a.binaries
                  if Path(entry[0]).name.lower() != 'icuuc.dll']
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MemeGenieAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=os.environ.get('MEMEGENIE_BUILD_CONSOLE') == '1',
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MemeGenieAI',
)
if sys.platform == 'darwin':
    app = BUNDLE(coll, name='MemeGenieAI.app',
                 bundle_identifier='com.memegenieai.studio')
