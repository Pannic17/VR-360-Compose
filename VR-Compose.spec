# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the distributed application. Build with tools/build_exe.py.

**Two shapes from one spec**, chosen by environment variables that the build script
sets. Run bare, it produces the onedir build -- the safe default:

* `VRC_ONEFILE=1` -- a single executable instead of a folder.
* `VRC_FFMPEG_DIR=<dir>` -- put that directory's ffmpeg.exe and ffprobe.exe *inside* the
  bundle. Only meaningful with `VRC_ONEFILE`: without it the folder build gets them
  copied in beside the executable, which is cheaper and does the same job.

One spec with one branch rather than two spec files, because the two shapes differ in
three lines and the failure mode of two files is that they drift.

`main_ui.py` is the entry point, and it is both the window and the CLI -- the frozen
window renders by launching the executable again with arguments. See its docstring.

## Which shape to ship (measured, AGENTS.md section 7)

**onedir is the better build and onefile is what the user asked for on 2026-09-08**,
having been shown these numbers. Both are supported; neither is a mistake to build.

| | onedir | onefile + bundled ffmpeg |
|---|---|---|
| what you copy | a 538 MiB folder | one 210 MiB file |
| `--version` warm | 0.35-0.39 s | seconds, see section 7 |
| per launch | nothing | unpacks the whole bundle to %TEMP% |

The onefile cost is structural, not a tuning problem: its bootloader unpacks the entire
archive on every launch, and ffmpeg is 370 MiB of that archive. There is no way to
unpack it once and keep it -- extraction happens before any of this project's code runs.

**No Qt exclude list**, in either shape. Adding 47 `--exclude-module` entries changed
the size by nothing at all: PyInstaller's PySide6 hook already ships only what is
imported. The work has been done once and is not worth repeating.
"""

import os
import pathlib

ROOT = pathlib.Path(SPECPATH)
GUI = ROOT / "src" / "vr_compose" / "gui"

ONEFILE = os.environ.get("VRC_ONEFILE") == "1"
FFMPEG_DIR = os.environ.get("VRC_FFMPEG_DIR") or None

# The window is built from these at runtime, so they are data and must be in the bundle;
# without them the exe starts and then cannot draw anything.
datas = [
    (str(GUI / "main_window.ui"), "vr_compose/gui"),
    (str(GUI / "style.qss"), "vr_compose/gui"),
    (str(GUI / "icon.png"), "vr_compose/gui"),  # the window's icon; the exe's is below
]

if FFMPEG_DIR:
    # As `datas`, not `binaries`: these are whole standalone programs the application
    # runs as subprocesses, not libraries it links, so there is nothing for PyInstaller's
    # dependency scanner to do with them but waste time. They land in the bundle root,
    # which is where `encode._search_dirs` looks (`sys._MEIPASS`).
    for name in ("ffmpeg.exe", "ffprobe.exe"):
        tool = pathlib.Path(FFMPEG_DIR) / name
        if not tool.is_file():
            raise SystemExit(f"VRC_FFMPEG_DIR is set but {name} is not in {FFMPEG_DIR}")
        datas.append((str(tool), "."))

analysis = Analysis(
    [str(ROOT / "main_ui.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=["vr_compose.cli", "vr_compose.gui.window"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # The GPU warp (`vr_compose.warp_gpu`) imports cupy lazily, but PyInstaller follows
    # imports inside functions too and its cupy hook would then drag ~1 GB of CUDA
    # libraries into a build meant for the GUI, which never asks for the GPU. Excluded on
    # purpose (approved 2026-09-09); `--device cuda` is for running from source.
    excludes=["cupy", "cupyx", "cupy_backends", "cuda", "fastrlock"],
    noarchive=False,
    optimize=0,
)

archive = PYZ(analysis.pure)

# **The console is the folder build's, and the single-file build has none.** This is one
# line with a measured reason on both sides (2026-09-09):
#
# *Single file* -- `console=False`. Its console belongs to the **bootloader** and is on
# screen before any of this project's code runs, so it cannot be hidden until the whole
# 494 MiB archive has been unpacked. A/B with a 70 MiB stand-in, sampling the desktop
# every 0.4 s: with a console and no hiding, a 1129x635 terminal window stayed up; with
# the hiding, the same window was visible about four seconds and then left a taskbar
# button; windowed, no console window appeared at any point. The real archive unpacks for
# far longer than four seconds, which is what the user reported as "it still pops up a
# command line window".
#
# *Folder* -- `console=True`, hidden by `main_ui._hide_own_console` for a double-click.
# It starts in 0.35 s and is a single process, so the console goes before it can be seen,
# and keeping it buys the thing a windowed build gives up: **PowerShell waits for a
# console process.** Measured on the windowed build, `VR-Compose.exe --version` typed bare
# into PowerShell prints nothing and sets no exit code, because the shell does not wait
# for a GUI-subsystem process; through a pipe, `cmd`, or `Start-Process -Wait` it is
# correct. So the shape people script against keeps its console, and the shape people
# double-click has none.
#
# What a windowed build does *not* lose, contrary to the note that used to be here: it
# does not leave `sys.stdout` as None whenever it is frozen. Windows attaches a new
# process to its parent's console whatever the subsystem says -- the subsystem only
# decides whether a *new* console is allocated when there is no parent one. Measured: the
# NDJSON pipe from `QProcess` arrives intact, which is what the window's progress needs.
# `main_ui._attach_parent_console` covers what is left, a start with no streams at all.
COMMON = dict(
    icon=str(GUI / "icon.ico"),  # the same picture as icon.png, in the sizes Explorer wants
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=not ONEFILE,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

if ONEFILE:
    executable = EXE(
        archive,
        analysis.scripts,
        analysis.binaries,
        analysis.datas,
        [],
        name="VR-Compose",
        runtime_tmpdir=None,
        **COMMON,
    )
else:
    executable = EXE(
        archive,
        analysis.scripts,
        [],
        exclude_binaries=True,
        name="VR-Compose",
        **COMMON,
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
