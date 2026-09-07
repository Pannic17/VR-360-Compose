"""Measure what a PySide6 + PyInstaller build actually costs, and check two traps.

Builds a throwaway app that imports exactly what the real one will (PySide6 widgets,
numpy, Pillow, a process pool, a subprocess ffmpeg call) and reports size and cold-start
time for each packaging strategy. Nothing here touches project or source data.

Two things this exists to catch:

* **A frozen process pool relaunching the app.** PyInstaller-frozen executables re-run
  `sys.executable` to spawn workers, so without `multiprocessing.freeze_support()` as the
  very first statement under `__main__`, every worker starts a fresh copy of the whole GUI
  -- recursively. P2's decode pool depends on this being right.
* **Qt's size.** PySide6 installs 632 MB. Most of that is WebEngine, Qt3D, Quick/QML,
  Multimedia and Charts, none of which this app needs.

    python tools/package_probe.py --configs excluded_onedir
    python tools/package_probe.py                              # all configs (slow)
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time

# The PySide6 subpackages this project will never import. They are the bulk of the 632 MB.
QT_EXCLUDES: tuple[str, ...] = (
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtQuick",
    "PySide6.QtQuick3D",
    "PySide6.QtQuickWidgets",
    "PySide6.QtQuickControls2",
    "PySide6.QtQml",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DRender",
    "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic",
    "PySide6.Qt3DAnimation",
    "PySide6.Qt3DExtras",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtGraphs",
    "PySide6.QtGraphsWidgets",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtSpatialAudio",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtDesigner",
    "PySide6.QtUiTools",
    "PySide6.QtHelp",
    "PySide6.QtSql",
    "PySide6.QtTest",
    "PySide6.QtBluetooth",
    "PySide6.QtNfc",
    "PySide6.QtPositioning",
    "PySide6.QtLocation",
    "PySide6.QtSerialPort",
    "PySide6.QtSerialBus",
    "PySide6.QtSensors",
    "PySide6.QtScxml",
    "PySide6.QtStateMachine",
    "PySide6.QtTextToSpeech",
    "PySide6.QtWebChannel",
    "PySide6.QtWebSockets",
    "PySide6.QtRemoteObjects",
    "PySide6.QtNetworkAuth",
    "PySide6.QtHttpServer",
    "PySide6.QtOpcUa",
    "PySide6.QtVirtualKeyboard",
    "PySide6.QtWebView",
    "tkinter",
    "scipy",
    "matplotlib",
    "pytest",
)

PROBE_APP = '''\
"""Throwaway probe app: same import surface as the real GUI will have."""

import multiprocessing
import sys


def _worker(n: int) -> int:
    import numpy as np

    return int(np.arange(n, dtype=np.int64).sum())


def selftest() -> int:
    import concurrent.futures
    import shutil
    import subprocess

    import numpy as np
    from PIL import Image
    from PySide6.QtCore import qVersion
    from PySide6.QtWidgets import QApplication, QLabel

    app = QApplication(sys.argv)
    label = QLabel("probe")
    checks = [f"qt={qVersion()}", f"numpy={np.__version__}", f"pil={Image.__version__}"]

    # The trap: a frozen exe respawns itself for pool workers. If freeze_support() were
    # missing, this line would start a new copy of the app instead of computing anything.
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(_worker, [1000, 2000]))
    checks.append(f"pool={results}")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        version = subprocess.run(
            [ffmpeg, "-hide_banner", "-version"], capture_output=True, text=True
        ).stdout.splitlines()[0]
        checks.append("ffmpeg=" + version.split()[2])
    else:
        checks.append("ffmpeg=NOT-ON-PATH")

    print("PROBE-OK " + " ".join(checks))
    label.deleteLater()
    app.quit()
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()  # MUST be first; see module docstring
    sys.exit(selftest())
'''


def _exclude_flags(modules: tuple[str, ...]) -> list[str]:
    flags: list[str] = []
    for module in modules:
        flags += ["--exclude-module", module]
    return flags


CONFIGS: dict[str, list[str]] = {
    "plain_onedir": ["--onedir"],
    "excluded_onedir": ["--onedir", *_exclude_flags(QT_EXCLUDES)],
    "excluded_onefile": ["--onefile", *_exclude_flags(QT_EXCLUDES)],
}


def dir_size(path: pathlib.Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def build(
    work: pathlib.Path, name: str, extra: list[str]
) -> tuple[pathlib.Path | None, float, str]:
    script = work / "probe_app.py"
    script.write_text(PROBE_APP, encoding="utf-8")
    dist, build_dir = work / f"dist_{name}", work / f"build_{name}"
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--name",
        name,
        "--distpath",
        str(dist),
        "--workpath",
        str(build_dir),
        "--specpath",
        str(work),
        *extra,
        str(script),
    ]
    started = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - started
    if result.returncode != 0:
        tail = " ".join((result.stderr or result.stdout).split())[-200:]
        return None, elapsed, tail
    exe = dist / f"{name}.exe"
    if not exe.exists():
        exe = dist / name / f"{name}.exe"
    return (exe if exe.exists() else None), elapsed, ""


def run_probe(exe: pathlib.Path) -> tuple[float, str]:
    started = time.time()
    result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=180)
    elapsed = time.time() - started
    line = next(
        (ln for ln in result.stdout.splitlines() if ln.startswith("PROBE-OK")),
        "NO PROBE-OK: " + " ".join((result.stderr or result.stdout).split())[-160:],
    )
    return elapsed, line


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument(
        "--keep",
        type=pathlib.Path,
        default=None,
        help="keep the build tree here instead of a temp directory",
    )
    args = parser.parse_args(argv)

    if shutil.which("ffmpeg") is None:
        print("note: ffmpeg is not on PATH, so the probe will report ffmpeg=NOT-ON-PATH")

    work = args.keep or pathlib.Path(tempfile.mkdtemp(prefix="vrc_pkg_"))
    work.mkdir(parents=True, exist_ok=True)
    print(f"build tree: {work}\n")
    print(f"{'config':<18} {'build':>7} {'bundle':>10} {'cold start':>11}  probe")
    failures = 0
    try:
        for name in args.configs:
            exe, build_time, error = build(work, name, CONFIGS[name])
            if exe is None:
                print(f"{name:<18} {build_time:6.0f}s  BUILD FAILED  {error}")
                failures += 1
                continue
            bundle = exe.parent if "onedir" in name else exe
            start, line = run_probe(exe)
            failures += not line.startswith("PROBE-OK")
            print(
                f"{name:<18} {build_time:6.0f}s {dir_size(bundle) / 2**20:9.0f}M "
                f"{start:10.2f}s  {line}"
            )
    finally:
        if args.keep is None:
            shutil.rmtree(work, ignore_errors=True)

    print(
        textwrap.dedent("""
        Reading this: `bundle` is what has to be shipped (the whole folder for onedir,
        the single file for onefile). Neither number includes ffmpeg -- the static
        ffmpeg.exe + ffprobe.exe on this machine are 185 MB each, so bundling both
        would dominate the download. `cold start` for onefile includes unpacking the
        archive to a temp directory on every launch.
    """)
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
