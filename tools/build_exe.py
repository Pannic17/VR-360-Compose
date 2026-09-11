"""Build the distributable: PyInstaller, then stage ffmpeg and the manual, then check it.

ROADMAP P8. The build has to be repeatable by one command, so everything the product
needs ends up in it here rather than in someone's memory.

    python tools/build_exe.py --ffmpeg C:/ffmpeg/bin              # the folder (default)
    python tools/build_exe.py --ffmpeg C:/ffmpeg/bin --onefile    # one executable
    python tools/build_exe.py --ffmpeg C:/ffmpeg/bin --gpu        # the folder, carrying CUDA
    python tools/build_exe.py --ffmpeg C:/ffmpeg/bin --gpu system --onefile  # one file, needs CUDA
    python tools/build_exe.py --skip-build                        # re-stage only
    python tools/build_exe.py --smoke --source E:/22              # + check what was built

**The smoke check is off by default** (user's instruction, 2026-09-08): a packaging or
documentation run should build the thing and stop. Turn it on with `--smoke` while
developing, which is where a failing check is worth the minutes it costs.

**Two shapes, and the difference is where ffmpeg goes.**

*onedir* (default) produces `dist/VR-Compose/`: the executable, `ffmpeg.exe` and
`ffprobe.exe` beside it, and the manual. `encode.find_tools()` looks beside the
executable before PATH, so the folder works on a machine with no ffmpeg installed --
which is the whole point of shipping it.

*onefile* (`--onefile`, user's decision on 2026-09-08) produces
`dist/onefile/VR-Compose.exe`: ffmpeg and ffprobe go *inside* the bundle, found at
runtime through `sys._MEIPASS`. That executable alone is the whole application -- copy
it anywhere and it runs. The manual is written beside it as documentation, not because
the program needs it.

The onefile cost is measured in AGENTS.md section 7 and is not a tuning problem: the
bootloader unpacks the entire archive on every launch, ffmpeg is most of that archive,
and extraction happens before any of this project's code could cache anything.

`--smoke` checks **what was built** rather than the source tree: a version query and,
given `--source`, a two-frame render whose output is compared byte for byte against the
same render from source. A mismatch fails the build.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = ROOT / "VR-Compose.spec"
DIST = ROOT / "dist"
MANUAL = ROOT / "docs" / "使用说明.md"
TOOLS = ("ffmpeg.exe", "ffprobe.exe")


@dataclasses.dataclass(frozen=True, slots=True)
class Layout:
    """Where this shape of build puts things, so nothing else has to branch on the mode."""

    onefile: bool

    @property
    def directory(self) -> pathlib.Path:
        """What gets zipped and handed over. For onefile that is just the exe's home."""
        return DIST / ("onefile" if self.onefile else "VR-Compose")

    @property
    def exe(self) -> pathlib.Path:
        return self.directory / "VR-Compose.exe"

    @property
    def describe(self) -> str:
        return "one executable" if self.onefile else "a self-contained folder"


def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    print(f"$ {' '.join(command)}", flush=True)
    return subprocess.run(command, text=True, **kwargs)  # type: ignore[call-overload,no-any-return]


def folder_size(path: pathlib.Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def locate_tools(source: pathlib.Path | None) -> list[pathlib.Path]:
    """The ffmpeg pair to ship, found the way the application would find them.

    Doing it here means a missing tool is reported by the build rather than by a user's
    first render, and it is the same list whether they get copied beside the executable
    or packed inside it.
    """
    found: list[pathlib.Path] = []
    for name in TOOLS:
        if source is not None:
            candidate = source / name
            if not candidate.is_file():
                raise SystemExit(f"{name} is not in {source}")
        else:
            located = shutil.which(name.removesuffix(".exe"))
            if located is None:
                raise SystemExit(
                    f"{name} was not found on PATH. Pass --ffmpeg with the directory "
                    "holding ffmpeg.exe and ffprobe.exe, or install ffmpeg."
                )
            candidate = pathlib.Path(located)
        found.append(candidate)
    return found


def ensure_not_running(exe: pathlib.Path) -> None:
    """Refuse to build over a copy of the app that is open.

    Windows locks a running executable's files, so PyInstaller's `--clean` fails when it
    tries to empty the directory -- and it fails deep inside `shutil.rmtree`, on whichever
    `.pyd` it reached first, which says nothing about the actual cause. Opening the
    executable for append asks the same question up front and gets the same answer.
    """
    if not exe.exists():
        return
    try:
        with exe.open("ab"):
            pass
    except PermissionError:
        raise SystemExit(
            f"{exe.name} is running, so its files cannot be replaced. Close the "
            "application (or its console) and run this again. Use --skip-build to "
            "re-stage ffmpeg and the manual without rebuilding."
        ) from None


def build(layout: Layout, tools: list[pathlib.Path], gpu: str | None = None) -> None:
    """Drive PyInstaller. The spec reads the shape out of the environment; see its docstring."""
    ensure_not_running(layout.exe)
    environment = dict(os.environ)
    # Set both ways round, always: a stale VRC_GPU in someone's shell would otherwise
    # decide the shape of a build that never asked for it, and it is a 600 MB decision.
    if gpu:
        environment["VRC_GPU"] = gpu
    else:
        environment.pop("VRC_GPU", None)
    if layout.onefile:
        environment["VRC_ONEFILE"] = "1"
        # The pair is already located and verified, so hand the spec their directory
        # rather than letting it search again and disagree.
        environment["VRC_FFMPEG_DIR"] = str(tools[0].parent)
    else:
        environment.pop("VRC_ONEFILE", None)
        environment.pop("VRC_FFMPEG_DIR", None)

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        # onefile writes `<distpath>/VR-Compose.exe`, while onedir's COLLECT creates a
        # `VR-Compose/` subdirectory of its own -- so the two want distpaths one level
        # apart to land in the same place.
        "--distpath",
        str(layout.directory if layout.onefile else DIST),
        str(SPEC),
    ]
    print(f"$ {' '.join(command)}", flush=True)
    result = subprocess.run(command, text=True, env=environment)
    if result.returncode != 0:
        raise SystemExit(f"PyInstaller failed with {result.returncode}")


def stage_ffmpeg(layout: Layout, tools: list[pathlib.Path]) -> None:
    if layout.onefile:
        print("    ffmpeg/ffprobe are inside the executable (VRC_FFMPEG_DIR)", flush=True)
        return
    for tool in tools:
        target = layout.directory / tool.name
        print(f"    staging {tool} -> {target}", flush=True)
        shutil.copy2(tool, target)


def stage_manual(layout: Layout) -> None:
    """Both the Markdown and a freshly printed PDF of it.

    The PDF is generated here rather than committed, so it cannot drift from the
    Markdown: there is no version of this build whose PDF is a release behind. Both ship
    because they are for different readers -- the PDF opens on any machine and prints,
    the Markdown is the one you edit and diff.

    For a onefile build these sit *beside* the executable as documentation. The
    executable does not need them, which is what makes it one file.
    """
    if not MANUAL.is_file():
        raise SystemExit(f"the manual is missing: {MANUAL}")
    layout.directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MANUAL, layout.directory / MANUAL.name)
    print(f"    staging {MANUAL.name}", flush=True)

    pdf = layout.directory / MANUAL.with_suffix(".pdf").name
    result = run([sys.executable, str(ROOT / "tools" / "manual_pdf.py"), "--out", str(pdf)])
    if result.returncode != 0 or not pdf.is_file():
        raise SystemExit(f"the manual PDF was not produced (exit {result.returncode})")
    print(f"    staging {pdf.name}  {pdf.stat().st_size / 1024:.0f} KiB", flush=True)


def digest(directory: pathlib.Path) -> list[str]:
    return [
        hashlib.sha256(item.read_bytes()).hexdigest()[:16]
        for item in sorted(directory.glob("*.png"))
    ]


def smoke(layout: Layout, source: pathlib.Path | None) -> None:
    """Exercise the built executable, not the source tree.

    The interesting one is the render: it proves the packaged app found its own ffmpeg --
    which for onefile means it found it inside itself -- that the warp plan and the
    encoder work frozen, and, because the output is compared against the source tree's,
    that freezing changed no pixels.

    The version query is timed twice on purpose. The first call and a later one differ by
    the whole cost of unpacking a onefile bundle, and that gap is the number someone
    deciding between the two shapes actually needs.
    """
    exe = layout.exe
    timings = []
    for _ in range(2):
        started = time.time()
        version = run([str(exe), "--version"], capture_output=True)
        timings.append(time.time() - started)
        if version.returncode != 0:
            raise SystemExit(f"the packaged executable failed --version:\n{version.stderr}")
    print(f"    --version -> {version.stdout.strip()!r}")
    print(f"    startup: {timings[0]:.2f} s first, {timings[1]:.2f} s again")

    if source is None:
        print("    (no --source given, skipping the render check)")
        return

    for label, target in (("packaged", DIST / "smoke_exe"), ("source", DIST / "smoke_src")):
        shutil.rmtree(target, ignore_errors=True)
        command = [str(exe)] if label == "packaged" else [sys.executable, str(ROOT / "main_ui.py")]
        started = time.time()
        result = run(
            [*command, "--source", str(source), "sequence", "--out-format", "png",
             "--frames", "1656-1657", "--sampler", "nearest", "--out", str(target),
             "--no-bar"],
            capture_output=True,
        )  # fmt: skip
        if result.returncode != 0:
            raise SystemExit(f"the {label} render failed:\n{result.stdout}\n{result.stderr}")
        print(f"    {label} render: {time.time() - started:.1f} s")

    packaged, from_source = digest(DIST / "smoke_exe"), digest(DIST / "smoke_src")
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
    parser.add_argument(
        "--onefile",
        action="store_true",
        help="one executable with ffmpeg inside it, instead of a folder (see the module docstring)",
    )
    parser.add_argument(
        "--gpu",
        nargs="?",
        const="bundled",
        choices=("bundled", "system"),
        default=None,
        help="build a GPU-capable executable. 'bundled' (the default when the flag is "
        "given bare) carries cupy and the CUDA libraries, so the target machine needs "
        "only an NVIDIA driver (+677 MB, folder shape only). 'system' carries cupy alone "
        "and uses the CUDA Toolkit 12.x on the target machine (+145 MB, either shape). "
        "See the spec's 'The GPU builds'",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run the checks against the built executable. Off by default (user's "
        "instruction, 2026-09-08): packaging and documentation runs should just build",
    )
    parser.add_argument("--skip-build", action="store_true", help="re-stage only")
    args = parser.parse_args(argv)

    if args.source is not None and not args.smoke:
        raise SystemExit(
            "--source is only used by the smoke check, which is off by default. Add "
            "--smoke to run it, or drop --source. Refusing rather than ignoring it, "
            "because a silently skipped render check looks exactly like a passing one."
        )

    if args.gpu == "bundled" and args.onefile:
        raise SystemExit(
            "--gpu bundled and --onefile do not go together. The single-file build "
            "unpacks its whole archive on every launch, and carrying the CUDA libraries "
            "adds 558 MB to what gets unpacked -- libraries the folder build pays for "
            "once, when it is copied. Use --gpu system for a single file: it expects a "
            "CUDA Toolkit 12.x on the target machine instead of carrying one."
        )

    layout = Layout(onefile=args.onefile)
    tools = locate_tools(args.ffmpeg)

    if not args.skip_build:
        build(layout, tools, gpu=args.gpu)
    if not layout.exe.is_file():
        raise SystemExit(f"no executable at {layout.exe}; run without --skip-build first")

    print("staging:", flush=True)
    stage_ffmpeg(layout, tools)
    stage_manual(layout)
    if args.smoke:
        print("smoke check:", flush=True)
        smoke(layout, args.source)

    print(f"\n{layout.directory}  {folder_size(layout.directory) / 2**20:.0f} MiB total")
    print(f"    {layout.exe.name}  {layout.exe.stat().st_size / 2**20:.0f} MiB")
    if layout.onefile:
        print("the executable alone is the whole application: copy it anywhere and run it.")
        print("the manual beside it is documentation; the program does not read it.")
    else:
        print("ready to zip. The folder is self-contained: unzip and run VR-Compose.exe.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
