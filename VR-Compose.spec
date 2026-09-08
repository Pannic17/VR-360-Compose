# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the distributed application. Build with tools/build_exe.py.

Two decisions here are measured rather than assumed (AGENTS.md section 7):

* **onedir, not onefile.** onefile is 64 MB against onedir's 158, and costs 5.88 s of
  warm start against 0.86. A 94 MB download is cheaper than five seconds on every launch
  of a tool someone uses all day.
* **No Qt exclude list.** Adding 47 `--exclude-module` entries changed the size by
  nothing at all: PyInstaller's PySide6 hook already ships only what is imported. The
  work has been done once and is not worth repeating.

`main_ui.py` is the entry point, and it is both the window and the CLI -- the frozen
window renders by launching the executable again with arguments. See its docstring.
"""

import pathlib

ROOT = pathlib.Path(SPECPATH)
GUI = ROOT / "src" / "vr_compose" / "gui"

analysis = Analysis(
    [str(ROOT / "main_ui.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    # The window is built from these at runtime, so they are data and must be in the
    # bundle; without them the exe starts and then cannot draw anything.
    datas=[
        (str(GUI / "main_window.ui"), "vr_compose/gui"),
        (str(GUI / "style.qss"), "vr_compose/gui"),
    ],
    hiddenimports=["vr_compose.cli", "vr_compose.gui.window"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

archive = PYZ(analysis.pure)

executable = EXE(
    archive,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="VR-Compose",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # A console is wanted: the window is only one of this executable's two faces, and the
    # other one is a command line whose output has to land somewhere. It also means a
    # crash on a user's machine leaves a readable traceback instead of a silent exit.
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

collection = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="VR-Compose",
)
