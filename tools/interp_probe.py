"""Frame interpolation: can a 30 fps render be taken to 60 fps offline? (ROADMAP P16, step 1)

A 60 fps render removes the trailing on moving content in the headset (upstream A11) at
twice the render, tiles and stitch time. This probe measures what an interpolator makes of
the same job, by **hold-out**: keep every other frame of a sequence, synthesise the dropped
ones, and compare with the frames that were really rendered. The truth is exact -- it is
the same render -- so nothing but the interpolator and the source's own coding is in the
numbers.

Everything works on directories of numbered RGB PNG frames (`000000.png`, ...):

    extract   a window of a video into such a directory
    stitch    a frame range of a tile source, through the product's PNG master and a
              Lanczos resize, into such a directory (the production path)
    holdout   keep every `--kept-step`-th frame of `--kept-dir`, synthesise the frames in
              between with each method, score them against `--truth-dir`
    tiles     interpolate each camera's tiles on their own and write the result as a tile
              source, to be stitched and scored (the "before the stitch" route)
    score     compare a directory of synthesised frames with `--truth-dir`

Synthesised frame `i` lies between kept frames `i` and `i + 1` and is compared with frame
`2 i + 1` of the truth directory, which is 60 fps numbering when the kept frames are 30 fps.
The three measurements of the plan are the same command with different directories:

    M1  60 fps video, hold-out      --kept-dir f60 --kept-step 2 --truth-dir f60
    M2  30 fps tiles, hold-out      --kept-dir s30 --kept-step 2 --truth-dir s30
    M3  30 fps video to 60 fps      --kept-dir f30 --kept-step 1 --truth-dir f60

Methods: `repeat` (the previous frame, the ceiling), `blend` (mean of the neighbours),
`minterpolate:<width>` (ffmpeg's block-matching interpolation at that working width) and
`rife:<model>[:pad]` (rife-ncnn-vulkan, given with `--rife`; `pad` wraps the equirect: the
left and right edges continue into each other, and a row past a pole is the row before it
seen from the other side, half a turn away).

All differences are absolute differences of Rec.709 luma in 8-bit levels:

    mean     cos(latitude)-weighted mean at 2048 x 1024, upstream pair_check.py's measure
    the rest at 3840 x 1920: plain means of the equator (|lat| < 30), the mid band (30-60),
    the zenith cap (lat > 60) and the nadir cap (lat < -60); the +-3 degree strip at the
    0/360 wrap against the rest, the per-frame 99.9th percentile, and 64 px block means
    (99th percentile and maximum over every block of every frame), those four within
    |lat| < 60.

The mean is dominated by the camera's own motion, which every interpolator handles; halos
and tears on fish, particles and occlusion edges cover little area and show in the block
maximum and the percentile. `--crops` saves the worst blocks side by side.

    python tools/interp_probe.py extract --video A11_A.mp4 --start 4 --seconds 1 --out w/f60
    python tools/interp_probe.py holdout --kept-dir w/f60 --kept-step 2 --truth-dir w/f60 \
        --work w/run --rife <rife-ncnn-vulkan dir> \
        --methods repeat blend minterpolate:3840 rife:rife-v4.6 rife:rife-v4.6:pad
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import pathlib
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from PIL import Image

from vr_compose import encode, source
from vr_compose.pipeline import parse_frames

F32 = npt.NDArray[np.float32]
U8 = npt.NDArray[np.uint8]

MEAN_WIDTH = 2048
"""The width upstream's `pair_check.py` compares at, so the two numbers mean the same."""
LOCAL_WIDTH = 3840
BLOCK = 64
WRAP_DEGREES = 3.0
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
RIFE_EXE = "rife-ncnn-vulkan.exe"


def frame_paths(directory: pathlib.Path) -> list[pathlib.Path]:
    paths = sorted(directory.glob("*.png"))
    if not paths:
        raise SystemExit(f"no PNG frames in {directory}")
    return paths


def read_rgb(path: pathlib.Path) -> U8:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def write_rgb(path: pathlib.Path, image: U8) -> None:
    Image.fromarray(image).save(path, format="PNG", compress_level=1)


def link(src: pathlib.Path, dst: pathlib.Path) -> None:
    """A hard link where the volume allows it, a copy where it does not."""
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copyfile(src, dst)


def fresh(directory: pathlib.Path) -> pathlib.Path:
    """Empty a directory this probe names itself, under a work directory it was given."""
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    return directory


def new_directory(directory: pathlib.Path) -> pathlib.Path:
    """A directory the caller named: created, never emptied -- it might be a tile source."""
    if directory.exists() and any(directory.iterdir()):
        raise SystemExit(f"{directory} is not empty; remove it or name another directory")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def run(command: Sequence[str | pathlib.Path], *, cwd: pathlib.Path | None = None) -> None:
    subprocess.run([str(part) for part in command], check=True, cwd=cwd)


# --------------------------------------------------------------------------- measuring


@dataclasses.dataclass(frozen=True, slots=True)
class Views:
    """One frame's luma at the two analysis sizes."""

    mean: F32
    local: F32


def views(path: pathlib.Path) -> Views:
    luma = Image.fromarray(read_rgb(path).astype(np.float32) @ LUMA)
    box = Image.Resampling.BOX
    return Views(
        mean=np.asarray(luma.resize((MEAN_WIDTH, MEAN_WIDTH // 2), box), dtype=np.float32),
        local=np.asarray(luma.resize((LOCAL_WIDTH, LOCAL_WIDTH // 2), box), dtype=np.float32),
    )


def _mean_weight() -> F32:
    height = MEAN_WIDTH // 2
    latitude = (0.5 - (np.arange(height) + 0.5) / height) * np.pi
    weight = np.cos(latitude)[:, None] / np.cos(latitude).sum() / MEAN_WIDTH
    return np.asarray(weight, dtype=np.float32)


MEAN_WEIGHT = _mean_weight()


def weighted_mean(a: F32, b: F32) -> float:
    return float((np.abs(a - b) * MEAN_WEIGHT).sum())


@dataclasses.dataclass(frozen=True, slots=True)
class Worst:
    value: float
    index: int
    row: int
    """Top row of the block at `LOCAL_WIDTH`, counted from the top of the frame."""
    column: int


class Stats:
    """Accumulates one method's differences over its frames."""

    HEAD = (
        f"{'method':<30s} {'mean':>6s}  {'<30':>6s} {'30-60':>6s} {'zenith':>6s} {'nadir':>6s}  "
        f"{'wrap':>6s} {'rest':>6s}  {'p99.9':>6s}  {'blk99':>6s} {'blkmax':>6s}"
    )

    def __init__(self) -> None:
        height = LOCAL_WIDTH // 2
        latitude = (0.5 - (np.arange(height) + 0.5) / height) * 180.0
        self._bands = (
            np.abs(latitude) < 30,
            (np.abs(latitude) >= 30) & (np.abs(latitude) < 60),
            latitude >= 60,
            latitude <= -60,
        )
        middle = np.flatnonzero(np.abs(latitude) < 60)
        self._top = int(middle[0])
        self._rows = (len(middle) // BLOCK) * BLOCK
        self._strip = round(WRAP_DEGREES / 360.0 * LOCAL_WIDTH)
        self.means: list[float] = []
        self.bands: list[tuple[float, ...]] = []
        self.seam: list[tuple[float, float]] = []
        self.tail: list[float] = []
        self.blocks: list[F32] = []
        self.worst: list[Worst] = []

    def add(self, index: int, got: Views, want: Views) -> None:
        self.means.append(weighted_mean(got.mean, want.mean))
        diff = np.abs(got.local - want.local)
        self.bands.append(tuple(float(diff[rows].mean()) for rows in self._bands))
        middle = diff[self._top : self._top + self._rows]
        strip = self._strip
        edge = np.concatenate([middle[:, :strip], middle[:, -strip:]], axis=1)
        self.seam.append((float(edge.mean()), float(middle[:, strip:-strip].mean())))
        self.tail.append(float(np.percentile(middle, 99.9)))
        blocks = middle.reshape(self._rows // BLOCK, BLOCK, LOCAL_WIDTH // BLOCK, BLOCK)
        block_means: F32 = np.asarray(blocks.mean(axis=(1, 3)), dtype=np.float32)
        self.blocks.append(block_means.ravel())
        row, column = np.unravel_index(int(block_means.argmax()), block_means.shape)
        self.worst.append(
            Worst(
                float(block_means.max()),
                index,
                self._top + int(row) * BLOCK,
                int(column) * BLOCK,
            )
        )

    def line(self, name: str) -> str:
        if not self.means:
            return f"{name:<30s} (no frames)"
        bands = np.mean(self.bands, axis=0)
        seam = np.mean(self.seam, axis=0)
        blocks = np.concatenate(self.blocks)
        return (
            f"{name:<30s} {np.mean(self.means):6.2f}  "
            f"{bands[0]:6.2f} {bands[1]:6.2f} {bands[2]:6.2f} {bands[3]:6.2f}  "
            f"{seam[0]:6.2f} {seam[1]:6.2f}  {np.mean(self.tail):6.1f}  "
            f"{np.percentile(blocks, 99):6.2f} {blocks.max():6.1f}"
        )


def save_crops(
    stats: Stats,
    got: dict[int, pathlib.Path],
    truth: dict[int, pathlib.Path],
    directory: pathlib.Path,
    label: str,
    count: int,
    size: int = 512,
) -> None:
    """The `count` worst blocks as truth | synthesised | 4 x difference, at full size."""
    directory.mkdir(parents=True, exist_ok=True)
    for rank, worst in enumerate(sorted(stats.worst, key=lambda w: -w.value)[:count]):
        want, have = read_rgb(truth[worst.index]), read_rgb(got[worst.index])
        if want.shape != have.shape:
            resized = Image.fromarray(have).resize(
                (want.shape[1], want.shape[0]), Image.Resampling.BICUBIC
            )
            have = np.asarray(resized, dtype=np.uint8)
        scale = want.shape[1] / LOCAL_WIDTH
        centre_y = (worst.row + BLOCK / 2) * scale
        centre_x = (worst.column + BLOCK / 2) * scale
        top = int(min(max(centre_y - size / 2, 0), want.shape[0] - size))
        left = int(min(max(centre_x - size / 2, 0), want.shape[1] - size))
        a = want[top : top + size, left : left + size]
        b = have[top : top + size, left : left + size]
        delta = np.abs(a.astype(np.int16) - b.astype(np.int16)) * 4
        strip = np.concatenate([a, b, np.clip(delta, 0, 255).astype(np.uint8)], axis=1)
        stem = re.sub(r"[^A-Za-z0-9.-]+", "_", label)
        name = f"{stem}_{rank}_frame{worst.index:04d}_{worst.value:.0f}.png"
        write_rgb(directory / name, strip)


# --------------------------------------------------------------------------- equirect padding


def pad_equirect(image: U8, wrap: int, pole: int) -> U8:
    """Continue an equirect past its edges the way the sphere does.

    Past the left edge is the right edge. Past a pole is the same meridian seen from the
    other side: the rows next to the pole, upside down and half a turn away.
    """
    width = image.shape[1]
    if pole:
        above = np.roll(image[:pole][::-1], width // 2, axis=1)
        below = np.roll(image[-pole:][::-1], width // 2, axis=1)
        image = np.concatenate([above, image, below], axis=0)
    if wrap:
        image = np.concatenate([image[:, -wrap:], image, image[:, :wrap]], axis=1)
    return image


def crop_equirect(image: U8, wrap: int, pole: int) -> U8:
    height, width = image.shape[0] - 2 * pole, image.shape[1] - 2 * wrap
    return image[pole : pole + height, wrap : wrap + width]


# --------------------------------------------------------------------------- video memory


def gpu_memory_used(gpu: int) -> int | None:
    """Memory in use on the card, MiB, or None without nvidia-smi."""
    try:
        text = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
             "-i", str(gpu)],
            capture_output=True, text=True, check=True,
        ).stdout  # fmt: skip
        return int(text.strip().splitlines()[0])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


class MemoryWatch:
    """Polls the card's memory while something else runs; reports the rise over the start."""

    def __init__(self, gpu: int) -> None:
        self._gpu = gpu
        self._stop = threading.Event()
        self._base = gpu_memory_used(gpu)
        self._peak = self._base
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self) -> None:
        while not self._stop.wait(0.2):
            used = gpu_memory_used(self._gpu)
            if used is not None and (self._peak is None or used > self._peak):
                self._peak = used

    def __enter__(self) -> MemoryWatch:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join()

    def report(self) -> str:
        if self._base is None or self._peak is None:
            return "video memory not measured"
        return f"video memory +{self._peak - self._base} MiB (card peak {self._peak} MiB)"


# --------------------------------------------------------------------------- methods


@dataclasses.dataclass(frozen=True, slots=True)
class Method:
    label: str
    kind: str
    width: int = 0
    model: str = ""
    pad: bool = False
    flags: tuple[str, ...] = ()


def parse_method(text: str) -> Method:
    kind, *rest = text.split(":")
    if kind in ("repeat", "blend") and not rest:
        return Method(text, kind)
    if kind == "minterpolate" and len(rest) == 1:
        return Method(text, kind, width=int(rest[0]))
    if kind == "rife" and rest:
        options = rest[1:]
        unknown = set(options) - {"pad", "x", "z", "u"}
        if unknown:
            raise SystemExit(f"unknown rife option(s) {sorted(unknown)} in {text!r}")
        flags = tuple(f"-{flag}" for flag in options if flag != "pad")
        return Method(text, kind, model=rest[0], pad="pad" in options, flags=flags)
    raise SystemExit(
        f"method {text!r}: expected repeat, blend, minterpolate:<width> or rife:<model>[:pad]"
    )


@dataclasses.dataclass(slots=True)
class Result:
    frames: dict[int, pathlib.Path]
    """Synthesised frame `i`, between kept frames `i` and `i + 1`."""
    note: str = ""


def do_blend(kept: Sequence[pathlib.Path], out: pathlib.Path) -> Result:
    frames: dict[int, pathlib.Path] = {}
    previous = read_rgb(kept[0]).astype(np.uint16)
    for i in range(len(kept) - 1):
        following = read_rgb(kept[i + 1]).astype(np.uint16)
        frames[i] = out / f"{i:06d}.png"
        write_rgb(frames[i], ((previous + following + 1) // 2).astype(np.uint8))
        previous = following
    return Result(frames)


def do_minterpolate(
    kept: Sequence[pathlib.Path], out: pathlib.Path, work: pathlib.Path, width: int
) -> Result:
    """ffmpeg's motion-compensated interpolation, kept frames at 30 fps in, 60 fps out."""
    tools = encode.find_tools()
    staged = fresh(work / "in")
    for i, path in enumerate(kept):
        link(path, staged / f"{i:06d}.png")
    raw = fresh(work / "raw")
    chain = f"scale={width}:{width // 2}:flags=area,minterpolate=fps=60:mi_mode=mci"
    started = time.time()
    try:
        run([tools.ffmpeg, "-v", "error", "-framerate", "30", "-start_number", "0",
             "-i", staged / "%06d.png", "-vf", chain, "-start_number", "0",
             raw / "%06d.png"])  # fmt: skip
    except subprocess.CalledProcessError:
        return Result({}, f"ffmpeg could not run minterpolate at {width} x {width // 2}")
    elapsed = time.time() - started
    # Output k is time k / 60: even k is a kept frame passed through, odd k is new. The
    # filter stops short of the last pair, so the count is taken from what it wrote.
    frames: dict[int, pathlib.Path] = {}
    for path in sorted(raw.glob("*.png")):
        k = int(path.stem)
        if k % 2 == 1 and k // 2 < len(kept) - 1:
            frames[k // 2] = out / f"{k // 2:06d}.png"
            path.replace(frames[k // 2])
    per_frame = elapsed / max(len(frames), 1)
    return Result(
        frames,
        f"{elapsed:.1f} s for {len(frames)} frames = {per_frame:.2f} s each "
        f"(single thread, PNG decode and encode included)",
    )


def run_rife(
    rife: pathlib.Path,
    model: str,
    staged: pathlib.Path,
    raw: pathlib.Path,
    *,
    gpu: int,
    jobs: str | None,
    flags: Sequence[str],
) -> tuple[float, str]:
    """One directory through rife-ncnn-vulkan. Returns (seconds, video memory report).

    For N inputs it writes 2 N files numbered from 1: file 2 i + 1 is input i again, file
    2 i + 2 the frame halfway to input i + 1, and the last two are the last input twice.
    """
    if not (rife / RIFE_EXE).is_file() or not (rife / model).is_dir():
        raise SystemExit(f"{rife} does not hold {RIFE_EXE} and a {model} model directory")
    command: list[str | pathlib.Path] = [
        rife / RIFE_EXE, "-i", staged, "-o", raw, "-m", rife / model, "-g", str(gpu), *flags,
    ]  # fmt: skip
    if jobs:
        command += ["-j", jobs]
    started = time.time()
    with MemoryWatch(gpu) as watch:
        subprocess.run([str(part) for part in command], check=True, cwd=rife, capture_output=True)
    return time.time() - started, watch.report()


def digest(directory: pathlib.Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.glob("*.png"))
    }


def do_rife(
    kept: Sequence[pathlib.Path],
    out: pathlib.Path,
    work: pathlib.Path,
    method: Method,
    args: argparse.Namespace,
) -> Result:
    if args.rife is None:
        raise SystemExit("a rife method needs --rife <directory of rife-ncnn-vulkan>")
    wrap, pole = (args.wrap, args.pole) if method.pad else (0, 0)
    staged = fresh(work / "in")
    started = time.time()
    for i, path in enumerate(kept):
        if method.pad:
            write_rgb(staged / f"{i:06d}.png", pad_equirect(read_rgb(path), wrap, pole))
        else:
            link(path, staged / f"{i:06d}.png")
    prepare = time.time() - started
    raw = fresh(work / "raw")
    elapsed, memory = run_rife(
        args.rife, method.model, staged, raw, gpu=args.gpu, jobs=args.jobs, flags=method.flags
    )
    pairs = len(kept) - 1
    note = (
        f"{elapsed:.1f} s for {pairs} pairs = {elapsed / pairs:.2f} s each (its own PNG "
        f"decode and encode included, and it writes every input again); {memory}"
    )
    if method.pad:
        note += f"; padded to {read_rgb(staged / '000000.png').shape[1::-1]} in {prepare:.0f} s"
    if args.twice:
        again = fresh(work / "raw2")
        run_rife(
            args.rife, method.model, staged, again, gpu=args.gpu, jobs=args.jobs,
            flags=method.flags,
        )  # fmt: skip
        first, second = digest(raw), digest(again)
        differing = sum(first[name] != second.get(name) for name in first)
        note += (
            "; a second run is byte-identical"
            if not differing
            else f"; a second run differs in {differing} of {len(first)} files"
        )
        shutil.rmtree(again)
    frames: dict[int, pathlib.Path] = {}
    for i in range(pairs):
        produced = raw / f"{2 * i + 2:08d}.png"
        frames[i] = out / f"{i:06d}.png"
        if method.pad:
            write_rgb(frames[i], crop_equirect(read_rgb(produced), wrap, pole))
        else:
            produced.replace(frames[i])
    shutil.rmtree(raw)
    shutil.rmtree(staged)
    return Result(frames, note)


# --------------------------------------------------------------------------- commands


def cmd_extract(args: argparse.Namespace) -> int:
    tools = encode.find_tools()
    out = new_directory(args.out)
    run([tools.ffmpeg, "-v", "error", "-ss", str(args.start), "-t", str(args.seconds),
         "-i", args.video, "-fps_mode", "passthrough", "-start_number", "0",
         out / "%06d.png"])  # fmt: skip
    print(f"{len(frame_paths(out))} frame(s) from {args.start:g} s of {args.video} in {out}")
    return 0


def cmd_stitch(args: argparse.Namespace) -> int:
    """Tiles -> the product's PNG master -> Lanczos to `--width`, numbered from 0."""
    tools = encode.find_tools()
    # Re-running continues the master (it resumes by file), so the directory may exist --
    # but only as this command left it: anything else in it is not ours to delete.
    if args.out.exists():
        foreign = [
            entry.name
            for entry in args.out.iterdir()
            if entry.name != "master" and not re.fullmatch(r"\d{6}\.png", entry.name)
        ]
        if foreign:
            raise SystemExit(f"{args.out} holds other files ({foreign[0]}, ...); name another")
    args.out.mkdir(parents=True, exist_ok=True)
    master = args.out / "master"
    command: list[str | pathlib.Path] = [
        sys.executable, "-m", "vr_compose", "--source", args.source, "sequence",
        "--frames", args.frames, "--out-format", "png", "--out", master,
        "--compress-level", "1", "--device", args.device, "--gate", "off",
        "--layout-gate", args.layout_gate,
    ]  # fmt: skip
    started = time.time()
    run(command)
    masters = sorted(master.glob("*_S_*.png"))
    if not masters:
        raise SystemExit(f"the stitch wrote no master into {master}")
    with Image.open(masters[0]) as first:
        native = first.width
    for stale in args.out.glob("*.png"):
        stale.unlink()
    if native == args.width:
        for i, path in enumerate(masters):
            link(path, args.out / f"{i:06d}.png")
    else:
        staged = fresh(args.out / "numbered")
        for i, path in enumerate(masters):
            link(path, staged / f"{i:06d}.png")
        run([tools.ffmpeg, "-v", "error", "-start_number", "0", "-i", staged / "%06d.png",
             "-vf", f"scale={args.width}:{args.width // 2}:flags=lanczos",
             "-start_number", "0", args.out / "%06d.png"])  # fmt: skip
        shutil.rmtree(staged)
    print(
        f"{len(masters)} frame(s), master {native} wide -> {args.width} wide, in {args.out} "
        f"({time.time() - started:.0f} s)"
    )
    return 0


def _truth(kept_count: int, truth_dir: pathlib.Path) -> dict[int, pathlib.Path]:
    truth = frame_paths(truth_dir)
    return {i: truth[2 * i + 1] for i in range(kept_count - 1) if 2 * i + 1 < len(truth)}


def cmd_holdout(args: argparse.Namespace) -> int:
    kept = frame_paths(args.kept_dir)[:: args.kept_step]
    if args.limit:
        kept = kept[: args.limit]
    truth = _truth(len(kept), args.truth_dir)
    if not truth:
        raise SystemExit("nothing to compare: the truth directory is too short")
    print(
        f"{len(kept)} kept frame(s) from {args.kept_dir} (every {args.kept_step}), "
        f"{len(truth)} synthesised frame(s) scored against {args.truth_dir}"
    )
    wanted = {i: views(path) for i, path in truth.items()}

    if args.cross_dir is not None:
        other = frame_paths(args.cross_dir)
        every = frame_paths(args.truth_dir)
        same: list[float] = []
        half: list[float] = []
        for j in range(min(len(other), len(every) // 2)):
            mine = views(other[j]).mean
            same.append(weighted_mean(mine, views(every[2 * j]).mean))
            half.append(weighted_mean(mine, views(every[2 * j + 1]).mean))
        print(
            f"other render, frame j against 2j: {np.mean(same):.2f} "
            f"({min(same):.2f}..{max(same):.2f}); against 2j+1: {np.mean(half):.2f} "
            f"({min(half):.2f}..{max(half):.2f})"
        )

    print(Stats.HEAD)
    for text in args.methods:
        method = parse_method(text)
        folder = args.work / text.replace(":", "_")
        out = fresh(folder / "frames")
        if method.kind == "repeat":
            result = Result({i: kept[i] for i in range(len(kept) - 1)})
        elif method.kind == "blend":
            result = do_blend(kept, out)
        elif method.kind == "minterpolate":
            result = do_minterpolate(kept, out, folder, method.width)
        else:
            result = do_rife(kept, out, folder, method, args)
        stats = Stats()
        for i in sorted(set(result.frames) & set(wanted)):
            stats.add(i, views(result.frames[i]), wanted[i])
        print(stats.line(method.label), flush=True)
        if result.note:
            print(f"    {result.note}", flush=True)
        if args.crops and stats.means:
            save_crops(stats, result.frames, truth, args.work / "crops", folder.name, args.crops)
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    got = dict(enumerate(frame_paths(args.got)))
    truth = _truth(len(got) + 1, args.truth_dir)
    stats = Stats()
    for i in sorted(set(got) & set(truth)):
        stats.add(i, views(got[i]), views(truth[i]))
    print(f"{len(stats.means)} frame(s) of {args.got} against {args.truth_dir}")
    print(Stats.HEAD)
    print(stats.line(args.label))
    if args.crops:
        save_crops(stats, got, truth, args.crops_dir or args.got / "crops", args.label, args.crops)
    return 0


def cmd_tiles(args: argparse.Namespace) -> int:
    """Each camera's tiles interpolated on their own, written as a tile source of the
    synthesised frames only.

    `--kept-step 2` is the hold-out: every other frame is kept and a synthesised frame
    takes the number of the frame it stands in for. `--kept-step 1` is the real job, every
    frame kept: there is no truth to score against, but whether the cameras still agree
    on where things are can be measured (`align_probe.py`), and the synthesised frame
    between `f` and `f + 1` is numbered `2 f + 1`, as a 60 fps render numbers it.
    """
    if args.rife is None:
        raise SystemExit("tiles needs --rife <directory of rife-ncnn-vulkan>")
    sets = [s for s in source.scan(args.source) if args.stem in (None, s.stem)]
    if len(sets) != 1 or not sets[0].usable or sets[0].tile_format != "png":
        raise SystemExit(f"{args.source}: expected one usable PNG tile set (pass --stem)")
    tiles = sets[0]
    frames = parse_frames(args.frames, tiles.frames)
    step: int = args.kept_step
    kept = frames[::step]
    if any(b - a != step for a, b in itertools.pairwise(kept)):
        raise SystemExit("--frames must be one run of consecutive frames")
    numbers = [frame + 1 if step == 2 else 2 * frame + 1 for frame in kept[:-1]]
    digits = max(tiles.frame_digits, len(str(numbers[-1])))
    work = fresh(new_directory(args.out) / "_work")
    seconds = 0.0
    memory = ""
    for camera in tiles.cameras:
        staged = fresh(work / "in")
        for i, frame in enumerate(kept):
            link(camera.path_for(tiles.stem, frame, tiles.frame_digits), staged / f"{i:06d}.png")
        raw = fresh(work / "raw")
        elapsed, memory = run_rife(
            args.rife, args.model, staged, raw, gpu=args.gpu, jobs=args.jobs, flags=()
        )
        seconds += elapsed
        directory = fresh(args.out / f"Camera{camera.index:02d}")
        for i, number in enumerate(numbers):
            name = f"{tiles.stem}.{number:0{digits}d}.png"
            (raw / f"{2 * i + 2:08d}.png").replace(directory / name)
        print(f"camera {camera.index:02d}: {elapsed:.1f} s", flush=True)
    shutil.rmtree(work)
    pairs = len(kept) - 1
    print(
        f"{len(tiles.cameras)} cameras x {pairs} frames in {seconds:.0f} s = "
        f"{seconds / pairs:.2f} s per panorama; {memory}; tile source in {args.out}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def rife_options(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--rife", type=pathlib.Path, default=None, help="rife-ncnn-vulkan's directory"
        )
        command.add_argument("--gpu", type=int, default=0, help="its -g: the Vulkan device")
        command.add_argument("--jobs", default=None, help="its -j load:proc:save thread counts")

    extract = sub.add_parser("extract", help="a window of a video as numbered PNG frames")
    extract.add_argument("--video", type=pathlib.Path, required=True)
    extract.add_argument("--start", type=float, required=True, help="seconds; use a keyframe")
    extract.add_argument("--seconds", type=float, default=1.0)
    extract.add_argument("--out", type=pathlib.Path, required=True)
    extract.set_defaults(run=cmd_extract)

    stitch = sub.add_parser("stitch", help="a tile source's frames as numbered PNG frames")
    stitch.add_argument("--source", type=pathlib.Path, required=True)
    stitch.add_argument("--frames", required=True, help="as vr-compose sequence --frames")
    stitch.add_argument("--out", type=pathlib.Path, required=True)
    stitch.add_argument("--width", type=int, default=7680, help="delivery width")
    stitch.add_argument("--device", default="auto")
    stitch.add_argument(
        "--layout-gate", choices=("on", "off"), default="on",
        help="off to stitch tiles the layout gate refuses, as independently interpolated "
        "ones are: their cameras no longer agree on where things are",
    )  # fmt: skip
    stitch.set_defaults(run=cmd_stitch)

    holdout = sub.add_parser("holdout", help="synthesise the dropped frames and score them")
    holdout.add_argument("--kept-dir", type=pathlib.Path, required=True)
    holdout.add_argument("--kept-step", type=int, default=2, choices=(1, 2))
    holdout.add_argument("--truth-dir", type=pathlib.Path, required=True)
    holdout.add_argument(
        "--cross-dir", type=pathlib.Path, default=None,
        help="the same window of another render at half the rate, for pair_check's floor "
        "and ceiling",
    )  # fmt: skip
    holdout.add_argument("--work", type=pathlib.Path, required=True)
    holdout.add_argument("--methods", nargs="+", default=["repeat", "blend"])
    holdout.add_argument("--limit", type=int, default=0, help="only the first N kept frames")
    holdout.add_argument("--wrap", type=int, default=256, help="columns of wrap for :pad")
    holdout.add_argument("--pole", type=int, default=128, help="rows past each pole for :pad")
    holdout.add_argument("--twice", action="store_true", help="run rife twice, compare bytes")
    holdout.add_argument("--crops", type=int, default=0, help="save the N worst blocks")
    rife_options(holdout)
    holdout.set_defaults(run=cmd_holdout)

    score = sub.add_parser("score", help="synthesised frames against the truth directory")
    score.add_argument("--got", type=pathlib.Path, required=True)
    score.add_argument("--truth-dir", type=pathlib.Path, required=True)
    score.add_argument("--label", default="synthesised")
    score.add_argument("--crops", type=int, default=0)
    score.add_argument("--crops-dir", type=pathlib.Path, default=None)
    score.set_defaults(run=cmd_score)

    tiles = sub.add_parser("tiles", help="interpolate every camera's tiles on their own")
    tiles.add_argument("--source", type=pathlib.Path, required=True)
    tiles.add_argument("--stem", default=None)
    tiles.add_argument("--frames", required=True, help="one run, e.g. 1300-1360")
    tiles.add_argument("--kept-step", type=int, default=2, choices=(1, 2))
    tiles.add_argument("--model", default="rife-v4.6")
    tiles.add_argument("--out", type=pathlib.Path, required=True)
    rife_options(tiles)
    tiles.set_defaults(run=cmd_tiles)

    args = parser.parse_args()
    result: int = args.run(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
