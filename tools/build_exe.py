"""Build the distributable folder: PyInstaller, then stage ffmpeg and the manual.

ROADMAP P8. The build has to be repeatable by one command, so everything the folder
needs ends up in it here rather than in someone's memory:

1. PyInstaller against `VR-Compose.spec` (onedir -- see the spec for why).
2. `ffmpeg.exe` and `ffprobe.exe` copied next to the executable. `encode.find_tools()`
   looks beside the executable *before* PATH, so a copied pair makes the folder work on
   a machine with no ffmpeg installed, which is the whole point of shipping it.
3. The Chinese manual copied in, because a folder someone unzips should explain itself.
4. A smoke check on what was built: version, discovery, and a real two-frame render, run
   through the packaged executable rather than the source tree.

    python tools/build_exe.py
    python tools/build_exe.py --skip-build          # re-stage and re-check only
    python tools/build_exe.py --ffmpeg C:/ffmpeg/bin
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = ROOT / "VR-Compose.spec"
DIST = ROOT / "dist" / "VR-Compose"
EXE = DIST / "VR-Compose.exe"
MANUAL = ROOT / "docs" / "使用说明.md"
TOOLS = ("ffmpeg.exe", "ffprobe.exe")


def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    print(f"$ {' '.join(command)}", flush=True)
    return subprocess.run(command, text=True, **kwargs)  # type: ignore[call-overload,no-any-return]


def folder_size(path: pathlib.Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def build() -> None:
    result = run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", str(SPEC)])
    if result.returncode != 0:
        raise SystemExit(f"PyInstaller failed with {result.returncode}")


def stage_ffmpeg(source: pathlib.Path | None) -> None:
    """Copy ffmpeg and ffprobe beside the executable.

    Located the same way the application will look for them if this step is skipped, so
    a missing tool is reported here rather than by a user's first render.
    """
    for name in TOOLS:
        if source is not None:
            found: pathlib.Path | None = source / name
            if not (found and found.is_file()):
                raise SystemExit(f"{name} is not in {source}")
        else:
            located = shutil.which(name.removesuffix(".exe"))
            if located is None:
                raise SystemExit(
                    f"{name} was not found on PATH. Pass --ffmpeg with the directory "
                    "holding ffmpeg.exe and ffprobe.exe, or install ffmpeg."
                )
            found = pathlib.Path(located)
        target = DIST / name
        print(f"    staging {found} -> {target}", flush=True)
        shutil.copy2(found, target)


def stage_manual() -> None:
    if not MANUAL.is_file():
        raise SystemExit(f"the manual is missing: {MANUAL}")
    shutil.copy2(MANUAL, DIST / MANUAL.name)
    print(f"    staging {MANUAL.name}", flush=True)


def smoke(source: pathlib.Path | None) -> None:
    """Exercise the built executable, not the source tree.

    The interesting one is the render: it proves the packaged app found its own ffmpeg,
    that the warp plan and the encoder work frozen, and -- because the output is compared
    against the source tree's -- that freezing changed no pixels.
    """
    started = time.time()
    version = run([str(EXE), "--version"], capture_output=True)
    print(f"    --version -> {version.stdout.strip()!r} in {time.time() - started:.2f} s")
    if version.returncode != 0:
        raise SystemExit(f"the packaged executable failed --version:\n{version.stderr}")

    if source is None:
        print("    (no --source given, skipping the render check)")
        return

    for label, target in (
        ("packaged", DIST.parent / "smoke_exe"),
        ("source", DIST.parent / "smoke_src"),
    ):
        shutil.rmtree(target, ignore_errors=True)
        command = [str(EXE)] if label == "packaged" else [sys.executable, str(ROOT / "main_ui.py")]
        result = run(
            [*command, "--source", str(source), "sequence", "--out-format", "png",
             "--frames", "1656-1657", "--sampler", "nearest", "--out", str(target),
             "--no-bar"],
            capture_output=True,
        )  # fmt: skip
        if result.returncode != 0:
            raise SystemExit(f"the {label} render failed:\n{result.stdout}\n{result.stderr}")

    import hashlib

    def digest(directory: pathlib.Path) -> list[str]:
        return [
            hashlib.sha256(item.read_bytes()).hexdigest()[:16]
            for item in sorted(directory.glob("*.png"))
        ]

    packaged, from_source = digest(DIST.parent / "smoke_exe"), digest(DIST.parent / "smoke_src")
    print(f"    packaged masters : {packaged}")
    print(f"    source masters   : {from_source}")
    if not packaged or packaged != from_source:
        raise SystemExit("the frozen build did not produce byte-identical output")
    print("    byte-identical to the source tree: OK")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ffmpeg",
        type=pathlib.Path,
        default=None,
        help="directory holding ffmpeg.exe and ffprobe.exe (default: look on PATH)",
    )
    parser.add_argument(
        "--source",
        type=pathlib.Path,
        default=None,
        help="a real source set, to render two frames through the packaged executable",
    )
    parser.add_argument("--skip-build", action="store_true", help="re-stage and re-check only")
    args = parser.parse_args(argv)

    if not args.skip_build:
        build()
    if not EXE.is_file():
        raise SystemExit(f"no executable at {EXE}; run without --skip-build first")

    print("staging:", flush=True)
    stage_ffmpeg(args.ffmpeg)
    stage_manual()
    print("smoke check:", flush=True)
    smoke(args.source)

    size = folder_size(DIST)
    print(f"\n{DIST}  {size / 2**20:.0f} MiB")
    print("ready to zip. The folder is self-contained: unzip and run VR-Compose.exe.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
