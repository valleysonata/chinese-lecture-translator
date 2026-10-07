# -*- mode: python ; coding: utf-8 -*-
"""
Reproducible PyInstaller spec for the Chinese Lecture Interpreter.

Build (from the repo root):
    py -m PyInstaller --noconfirm --clean ChineseLectureInterpreter.spec
or:
    .\\build_exe.ps1

Result: dist\\ChineseLectureInterpreter\\ChineseLectureInterpreter.exe
(windowed — no terminal required; PyQt6 UI, mic, PDF, ASR/translation all bundled).

SECURITY: the API key is NEVER bundled. The frozen app loads .env from the
directory next to the exe (see config.py frozen branch) and storage/ is
created there too.
"""
from PyInstaller.utils.hooks import collect_data_files

datas = []
# PortAudio shared libraries ship in a separate top-level data package
datas += collect_data_files("_sounddevice_data")
# Silero VAD model files (.jit / .onnx / .safetensors)
datas += collect_data_files("silero_vad")

hiddenimports = [
    # scripts/ is an implicit namespace package; make the shared console
    # helpers import (from scripts.test_phase3 import ...) explicit for the
    # frozen module graph
    "scripts.test_phase3",
]

a = Analysis(
    ["scripts/app.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ChineseLectureInterpreter",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # windowed: launches without a terminal
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
    upx=False,
    name="ChineseLectureInterpreter",
)
