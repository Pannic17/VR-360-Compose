"""Build the distributable: PyInstaller, then stage ffmpeg, the interpolator and the manual.

ROADMAP P8. The build has to be repeatable by one command, so everything the product
needs ends up in it here rather than in someone's memory.

    python tools/build_exe.py --ffmpeg C:/ffmpeg/bin              # the folder (default)
    python tools/build_exe.py --ffmpeg C:/ffmpeg/bin --onefile    # one executable
    python tools/build_exe.py --ffmpeg C:/ffmpeg/bin --gpu        # the folder, carrying CUDA
    python tools/build_exe.py --ffmpeg C:/ffmpeg/bin --gpu system --onefile  # one file, needs CUDA
    python tools/build_exe.py --rife D:/tools/rife-ncnn-vulkan    # the interpolator from there
    python tools/build_exe.py --no-rife                           # a package without it
    python tools/build_exe.py --skip-build                        # re-stage only
    python tools/build_exe.py --smoke --source <a capture>        # + check what was built

**The smoke check is off by default** (user's instruction, 2026-09-08): a packaging or
documentation run should build the thing and stop. Turn it on with `--smoke` while
developing, which is where a failing check is worth the minutes it costs.

**The package carries the interpolator** (user's decision, 2026-11-11, ROADMAP P16):
rife-ncnn-vulkan and its `rife-v4.6` model, 17 MiB, as a `rife-ncnn-vulkan` folder beside
the executable or inside a onefile bundle -- the two places `interpolate.find` looks. It
is taken from `--rife`, or from wherever a run from source finds it (`bin/rife-ncnn-vulkan`
in this tree, or `VR_COMPOSE_RIFE`). A build that cannot find it stops, as one without
ffmpeg does, because a package that silently lacks it delivers 30 fps to everyone who
ticks the box; `--no-rife` builds one without it on purpose.

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
same render from source. A mismatch fails the build. The two frames come from the source
(`smoke_frames`), so any capture will do.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
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
RIFE_FOLDER = "rife-ncnn-vulkan"
"""The interpolator's folder in a package -- the name `vr_compose.interpolate` looks for."""
RIFE_EXE = "rife-ncnn-vulkan.exe"
RIFE_MODEL = "rife-v4.6"
RIFE_PAYLOAD = ROOT / "build" / RIFE_FOLDER
"""Where the files that ship are gathered before the build, so both shapes carry one list."""


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


def locate_rife(source: pathlib.Path | None) -> pathlib.Path:
    """The interpolator's folder to ship: `source`, or where a run from source finds it."""
    if source is None:
        from vr_compose import interpolate  # only this lookup needs the package

        found = interpolate.find()
        if found is None:
            raise SystemExit(
                f"the interpolator was not found ({interpolate.MISSING}). Pass --rife with "
                f"the folder holding {RIFE_EXE} and {RIFE_MODEL}/, or --no-rife to build a "
                "package without frame interpolation."
            )
        return found.directory
    if not (source / RIFE_EXE).is_file() or not (source / RIFE_MODEL).is_dir():
        raise SystemExit(f"{source} does not hold {RIFE_EXE} and a {RIFE_MODEL} folder")
    return source


def rife_files(directory: pathlib.Path) -> list[tuple[pathlib.Path, str]]:
    """`(file, subfolder)` for what ships: the executable, the libraries beside it, its
    licence, and the one model the product uses.

    The release it comes from carries eleven models and 448 MiB; the product runs one of
    them. The licence goes along because the MIT terms ask for it in every copy.
    """
    files = [(directory / RIFE_EXE, "")]
    files += [(library, "") for library in sorted(directory.glob("*.dll"))]
    files += [(notice, "") for notice in (directory / "LICENSE",) if notice.is_file()]
    files += [
        (weight, RIFE_MODEL)
        for weight in sorted((directory / RIFE_MODEL).iterdir())
        if weight.is_file()
    ]
    return files


def gather_rife(directory: pathlib.Path, payload: pathlib.Path = RIFE_PAYLOAD) -> pathlib.Path:
    """Copy what ships into `payload`, laid out as it will sit in the package."""
    shutil.rmtree(payload, ignore_errors=True)
    for file, subfolder in rife_files(directory):
        target = payload / subfolder
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target / file.name)
    return payload


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


def build(
    layout: Layout,
    tools: list[pathlib.Path],
    gpu: str | None = None,
    rife: pathlib.Path | None = None,
) -> None:
    """Drive PyInstaller. The spec reads the shape out of the environment; see its docstring."""
    ensure_not_running(layout.exe)
    environment = dict(os.environ)
    # Inside the bundle only for onefile, like ffmpeg: the folder build gets it copied in
    # beside the executable afterwards, which is cheaper and does the same job.
    if layout.onefile and rife is not None:
        environment["VRC_RIFE_DIR"] = str(rife)
    else:
        environment.pop("VRC_RIFE_DIR", None)
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


def stage_rife(layout: Layout, payload: pathlib.Path | None) -> None:
    if payload is None:
        print("    no interpolator (--no-rife): --interpolate 2x will warn and deliver 30 fps")
        return
    if layout.onefile:
        print("    the interpolator is inside the executable (VRC_RIFE_DIR)", flush=True)
        return
    target = layout.directory / RIFE_FOLDER
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(payload, target)
    print(f"    staging {RIFE_FOLDER}/  {folder_size(target) / 2**20:.0f} MiB", flush=True)


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


def smoke_frames(source: pathlib.Path) -> tuple[str, str]:
    """The stem and two frames of `source` for the render check, as `--stem` and `--frames` want.

    The first source set with no problems, and its first two consecutive frames every camera
    holds -- or its first two frames when none are consecutive, since a comma list selects
    exactly those. Read from the source, so the check runs on whatever capture is at hand.
    """
    from vr_compose.source import scan  # only the render check needs the package

    for found in scan(source):
        frames = found.frames
        if found.problems or len(frames) < 2:
            continue
        pair = next(
            ((a, b) for a, b in itertools.pairwise(frames) if b == a + 1),
            (frames[0], frames[1]),
        )
        return found.stem, f"{pair[0]},{pair[1]}"
    raise SystemExit(f"{source} holds no source set with two frames shared by every camera")


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

    stem, frames = smoke_frames(source)
    print(f"    render check: stem {stem!r}, frames {frames}")
    for label, target in (("packaged", DIST / "smoke_exe"), ("source", DIST / "smoke_src")):
        shutil.rmtree(target, ignore_errors=True)
        command = [str(exe)] if label == "packaged" else [sys.executable, str(ROOT / "main_ui.py")]
        started = time.time()
        result = run(
            [*command, "--source", str(source), "sequence", "--out-format", "png",
             "--stem", stem, "--frames", frames, "--sampler", "nearest", "--out", str(target),
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
        "only an NVIDIA driver, 50-series cards included (+882 MiB, folder shape only). "
        "'system' carries cupy alone and uses the CUDA Toolkit on the target machine, "
        "which must be 12.x and 12.8+ for a 50-series card (+145 MB, either shape). "
        "See the spec's 'The GPU builds'",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run the checks against the built executable. Off by default (user's "
        "instruction, 2026-09-08): packaging and documentation runs should just build",
    )
    parser.add_argument("--skip-build", action="store_true", help="re-stage only")
    parser.add_argument(
        "--rife",
        type=pathlib.Path,
        default=None,
        help=f"folder holding {RIFE_EXE} and {RIFE_MODEL}/ to ship as the interpolator "
        f"(default: where a run from source finds it -- bin/{RIFE_FOLDER}, or VR_COMPOSE_RIFE)",
    )
    parser.add_argument(
        "--no-rife",
        action="store_true",
        help="build a package without the interpolator; --interpolate 2x then warns and "
        "delivers at the capture's rate",
    )
    args = parser.parse_args(argv)

    if args.no_rife and args.rife is not None:
        raise SystemExit("--rife and --no-rife contradict each other; pass one")

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
            "adds 770 MB to what gets unpacked -- libraries the folder build pays for "
            "once, when it is copied. Use --gpu system for a single file: it expects a "
            "CUDA Toolkit 12.x on the target machine instead of carrying one."
        )

    layout = Layout(onefile=args.onefile)
    tools = locate_tools(args.ffmpeg)
    payload = None if args.no_rife else gather_rife(locate_rife(args.rife))

    if not args.skip_build:
        build(layout, tools, gpu=args.gpu, rife=payload)
    if not layout.exe.is_file():
        raise SystemExit(f"no executable at {layout.exe}; run without --skip-build first")

    print("staging:", flush=True)
    stage_ffmpeg(layout, tools)
    stage_rife(layout, payload)
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
