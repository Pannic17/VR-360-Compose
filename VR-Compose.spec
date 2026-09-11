# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the distributed application. Build with tools/build_exe.py.

**Two shapes from one spec**, chosen by environment variables that the build script
sets. Run bare, it produces the onedir build -- the safe default:

* `VRC_ONEFILE=1` -- a single executable instead of a folder.
* `VRC_FFMPEG_DIR=<dir>` -- put that directory's ffmpeg.exe and ffprobe.exe *inside* the
  bundle. Only meaningful with `VRC_ONEFILE`: without it the folder build gets them
  copied in beside the executable, which is cheaper and does the same job.
* `VRC_GPU=1` -- put cupy and the CUDA libraries into the bundle, so `--device cuda`
  works on a machine that has an NVIDIA driver and nothing else installed. Costs about
  680 MB; see "The GPU build" below. Folder shape only.

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

## The GPU builds (`VRC_GPU`, added 2026-09-10)

Without it the bundle has no cupy and `--device cuda` can never be served: **a frozen
build imports only what is inside it**. cupy installed on the target machine is
invisible, and so is `PYTHONPATH` -- measured 2026-09-11, pointing it straight at a
site-packages holding cupy, on a machine with the toolkit and a 4090: the exe still said
`ModuleNotFoundError: No module named 'cupy'`. There is no environment a user can
configure that turns a cupy-less build into a GPU one. It has to be built in.

**Two ways to build it in**, and they differ only in whether the toolkit rides along:

| `VRC_GPU` | carries | the target machine needs | shapes |
|---|---|---|---|
| `bundled` | cupy + the three CUDA libraries + the toolkit headers | an NVIDIA driver, nothing else | folder |
| `system` | cupy | **CUDA Toolkit 12.x installed**, plus the driver | folder or single file |

`system` is 558 MB smaller, which is what makes a single-file GPU build tolerable at all:
the bootloader unpacks the whole archive on every launch, so 558 MB of libraries that the
target already has on disk would be paid for again at every start. `bundled` asks nothing
of the target machine and is the one to hand to someone whose setup you do not control.

Neither one changes the fallback: a machine that cannot serve the GPU gets the CPU, with a
warning for `--device cuda` and silently for the `auto` the GUI sends (`device`, rule 4).
So a `system` build handed to a machine with no toolkit is not broken, just slower.

**What a `bundled` build has to carry, measured rather than assumed** (2026-09-10,
tracing the loaded modules of `import cupy` -> `cp.zeros` -> `RawKernel` on a 4090).
A `system` build carries only the cupy row; the rest comes from the target's toolkit:

| | | |
|---|---|---|
| `cublasLt64_12.dll` | 473 MB | loaded eagerly by `import cupy` |
| `nvrtc64_120_0.dll` | 45 MB | same -- kernels are compiled at runtime |
| `nvrtc-builtins64_124.dll` | 6 MB | on the first `RawKernel` compile |
| the cupy package | 158 MB | including 22 MB of headers NVRTC reads |
| the toolkit's `include/` | 35 MB | NVRTC reads those too, at every launch |

A first build carried 2525 MiB, because PyInstaller's dependency analysis imports what it
collects and follows every library those imports touch. What it collects that way is
dropped again below `Analysis`; see the comment there.

**Nothing else.** cublas, cufft, cusparse, cusolver, curand and nvJitLink are 1.4 GB
between them and not one of them is loaded -- the note that used to be here guessed
"~1 GB of CUDA libraries" and was both too high and about the wrong libraries.
`nvcuda.dll` is the *driver's*, lives in System32, and must never be copied: it belongs
to whatever card the machine has.

**A `bundled` bundle is a CUDA toolkit, as far as cupy is concerned**: `bin/` holds the
three libraries, `include/` holds the headers, and `tools/runtime_hook_cuda.py` points
`CUDA_PATH` at it. That is not decoration, it is the layout every lookup expects. A
`system` build ships none of that and sets nothing, so every one of those lookups lands
on the target's own `CUDA_PATH` -- which its toolkit installer set system-wide.

**The libraries go in `bin/`, and the bundle root must have none of them.** cupy works
out where the CUDA toolkit is by asking cuda-pathfinder for nvrtc and taking the
grandparent directory of the answer, then adds `<that>/bin` to the DLL search path
without checking it exists. A `bin/` subdirectory is therefore not decoration: it is the
layout that makes the arithmetic come out at the bundle. `tools/runtime_hook_cuda.py`
puts that directory on the DLL search path before anything imports cupy, which is how
pathfinder finds it in the first place -- PyInstaller only searches the bundle root.

**`bundled` is folder-shape only.** A onefile build already unpacks 494 MiB on every
launch; adding 680 MB to that is not a build anyone would wait for. `system` adds about
145 MB and is allowed in either shape.

**No Qt exclude list**, in either shape. Adding 47 `--exclude-module` entries changed
the size by nothing at all: PyInstaller's PySide6 hook already ships only what is
imported. The work has been done once and is not worth repeating.
"""

import os
import pathlib

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = pathlib.Path(SPECPATH)
GUI = ROOT / "src" / "vr_compose" / "gui"

ONEFILE = os.environ.get("VRC_ONEFILE") == "1"
FFMPEG_DIR = os.environ.get("VRC_FFMPEG_DIR") or None
GPU = os.environ.get("VRC_GPU") or ""
if GPU not in ("", "bundled", "system"):
    raise SystemExit(f"VRC_GPU must be 'bundled' or 'system', got {GPU!r}")

CUDA_LIBRARIES = ("cublasLt64_12.dll", "nvrtc64_120_0.dll", "nvrtc-builtins64_124.dll")
"""The three the warp actually loads, and they go in `bin/`. The docstring says why."""

CUDA_COLLECTED = (
    "cublas64_12.dll",
    "cublaslt64_12.dll",
    "cufft64_11.dll",
    "curand64_10.dll",
    "cusolver64_11.dll",
    "cusparse64_12.dll",
    "nvjitlink_120_0.dll",
    "nvrtc64_120_0.dll",
)
"""What PyInstaller puts in the bundle root by itself. All of it goes; the docstring says why."""

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

binaries = []
hiddenimports = ["vr_compose.cli", "vr_compose.gui.window"]
runtime_hooks = []
# The GPU warp (`vr_compose.warp_gpu`) imports cupy lazily, but PyInstaller follows
# imports inside functions too, so without this list a default build would carry the
# whole CUDA stack for a GUI that never asks for the GPU. Excluded unless VRC_GPU says
# otherwise (2026-09-09; the flag came 2026-09-10).
excludes = ["cupy", "cupyx", "cupy_backends", "cuda", "fastrlock"]

if GPU:
    excludes = []
    # cupy reaches several of its own submodules dynamically (`cupy.cuda.thrust`,
    # `cupy.fft._callback`), which a static analysis does not see.
    hiddenimports += collect_submodules("cupy")
    hiddenimports += collect_submodules("cupy_backends")
    hiddenimports += ["cupyx", "cuda.pathfinder"]
    # `graphlib` is imported by a *compiled* module in `cupy._core`, so nothing in any
    # source file mentions it and PyInstaller's analysis cannot see it. Without it
    # `import cupy` raises ModuleNotFoundError, which the device gate reads as "no GPU".
    # Found by diffing what `import cupy` loads against what the build contains; it was
    # the only real absence, and the only other candidate (`_cython_3_2_4`) is a module
    # Cython synthesises at import time rather than a file anyone can bundle.
    hiddenimports += ["graphlib"]
    # cupy's own headers: 22 MB of `.cuh` that NVRTC reads when it compiles a kernel.
    # They are data rather than imports, so nothing collects them by itself, and without
    # them the exe imports cupy happily and dies on the first `cp.zeros`.
    datas += collect_data_files("cupy")
if GPU == "bundled":
    cuda_root = pathlib.Path(os.environ.get("CUDA_PATH", ""))
    for name in CUDA_LIBRARIES:
        library = cuda_root / "bin" / name
        if not library.is_file():
            raise SystemExit(f"VRC_GPU=bundled needs {name}; CUDA_PATH is {os.environ.get('CUDA_PATH')!r}")
        binaries.append((str(library), "bin"))
    # The toolkit's headers, whole. NVRTC compiles the warp's kernels at runtime and
    # reads them, and cuda-pathfinder looks under `$CUDA_PATH/include` -- failing which
    # it goes looking for a CUDA installation by running `sys.executable -m ...`, which
    # in a frozen application is this executable, which reads that as command-line
    # arguments and exits. 35 MB, against picking headers by hand and finding out at
    # someone's first render which transitive include was missed.
    headers = cuda_root / "include"
    if not headers.is_dir():
        raise SystemExit(f"VRC_GPU=bundled needs the toolkit headers at {headers}")
    for header in headers.rglob("*"):
        if header.is_file():
            datas.append((str(header), str("include" / header.parent.relative_to(headers))))
    runtime_hooks.append(str(ROOT / "tools" / "runtime_hook_cuda.py"))

analysis = Analysis(
    [str(ROOT / "main_ui.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=runtime_hooks,
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

if GPU:
    # **Everything CUDA that PyInstaller collected by itself is dropped** -- in both
    # modes. A `bundled` build keeps the three it put in `bin/` itself; a `system` build
    # keeps none and uses the target's. Two separate reasons, both measured on 2026-09-10:
    #
    # *It is 884 MB of dead weight.* Importing every cupy submodule reaches the wrappers
    # for cuFFT, cuSPARSE, cuSOLVER, cuBLAS, cuRAND and nvJitLink, so the analysis brings
    # their libraries too. The warp is RawKernel, RawModule, asarray, zeros and asnumpy;
    # none of the six is ever loaded. The wrapper *modules* stay -- cupy imports them
    # itself, and only opens a library when someone does linear algebra, which nothing here
    # does. That is also why this is a filter and not an `excludes` entry.
    #
    # *The root copies break the import outright.* `cupy._environment` locates the toolkit
    # by asking cuda-pathfinder for nvrtc and taking `dirname(dirname(...))` of the answer,
    # then calls `os.add_dll_directory(root + "/bin")` without checking that it exists. Found
    # at the bundle root, that computes the *application* directory, whose `bin` does not
    # exist, and `import cupy` raises FileNotFoundError -- which the device gate reads as
    # "no GPU" and quietly runs on the CPU. Found in `bin/`, the same arithmetic gives the
    # bundle directory, whose `bin` is exactly where these libraries are. This is why the
    # filter is not conditional on the mode: a `system` build with a stray nvrtc at its
    # root would compute a root that is a *temporary unpack directory* and break the same
    # way, while having a perfectly good toolkit on the machine to use instead.
    analysis.binaries = [
        entry for entry in analysis.binaries if entry[0].lower() not in CUDA_COLLECTED
    ]

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
