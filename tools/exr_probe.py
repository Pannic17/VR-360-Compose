"""P15: the EXR source against the PNG source, tile by tile and stitched.

    python tools/exr_probe.py check  <capture>                  reader vs ffmpeg, decode timing
    python tools/exr_probe.py tiles  <capture A> <capture B>    per-tile bias, d = B - A, in levels
    python tools/exr_probe.py stitch <capture A> <capture B> [--frames 571,576] [--device cuda]
    python tools/exr_probe.py master <EXR master dir> [--png16 <16-bit PNG master dir>]

A capture is a directory `vr_compose.source` accepts (`Camera01..15` under it). `tiles` and
`stitch` pair the frames the two captures share; either may be PNG or EXR, so PNG/EXR,
PNG/PNG and EXR/EXR pairs all work -- two captures of the same process state drift, and the
PNG/PNG pair is what that drift looks like. `stitch` also builds the same-content control:
the EXR capture's float levels against those levels quantised with an unbiased dither (a
synthetic PNG of the very same picture), which is what "within quantisation" is measured
against. `master` checks an EXR master directory (`--out-format exr`, step 3): each file
against ffmpeg's decoder bit for bit, its bytes per pixel, and -- given the 16-bit PNG
master of the same run -- that it is that master's codes through the table, and how far
the half values decode back from the PNG's levels. Numbers in ROADMAP "P15".
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from vr_compose import encode, exr, harmonise, io, source, verify
from vr_compose.pipeline import Warper
from vr_compose.rig import Rig, rig_for
from vr_compose.stitch import Tile
from vr_compose.warp import WarpPlan

F32 = npt.NDArray[np.float32]
U8 = npt.NDArray[np.uint8]
U16 = npt.NDArray[np.uint16]

BINS = (0, 1, 4, 16, 64, 256)


def capture(path: pathlib.Path) -> source.SourceSet:
    found = [s for s in source.scan(path) if s.usable]
    if len(found) != 1:
        rejected = "\n".join(f"  {s.stem!r}: {s.problems}" for s in source.scan(path))
        raise SystemExit(f"{path}: expected one usable source set, found {len(found)}\n{rejected}")
    return found[0]


def levels(tile: Tile) -> F32:
    """Any tile as float32 levels: PNG tiles are integers, EXR tiles already float."""
    return np.asarray(tile, dtype=np.float32)


def diff_stats(a: io.Panorama, b: io.Panorama, scale: float) -> str:
    d = (b.astype(np.float64) - a.astype(np.float64)) / scale
    ad = np.abs(d)
    over = np.mean(ad > 1.0) * 100
    p99, p999 = np.percentile(ad, 99), np.percentile(ad, 99.9)
    return (
        f"bias {d.mean():+.4f}  mean|d| {ad.mean():.4f}  >1 level {over:.3f}%  "
        f"p99 {p99:.2f}  p99.9 {p999:.2f}  max {ad.max():.1f}"
    )


# -- check ---------------------------------------------------------------------------------


def ffmpeg_half(tools: encode.Tools, path: pathlib.Path, shape: tuple[int, int]) -> U16:
    """ffmpeg's own EXR decoder, half bits untouched: the independent reading."""
    command = [str(tools.ffmpeg), "-v", "error", "-i", str(path)]
    command += ["-f", "rawvideo", "-pix_fmt", "gbrapf16le", "-"]
    raw = subprocess.run(command, capture_output=True, check=True).stdout
    g, b, r, _alpha = np.frombuffer(raw, np.uint16).reshape(4, *shape)
    return np.stack([r, g, b], axis=-1)


def cmd_check(args: argparse.Namespace) -> int:
    chosen = capture(args.capture)
    if chosen.tile_format != "exr":
        raise SystemExit(f"{args.capture} is a {chosen.tile_format} capture; `check` reads EXR")
    tools = encode.find_tools()
    rig = rig_for(chosen.camera_count)
    frame = chosen.frames[0]
    print(f"capture    : {chosen.root}  stem {chosen.stem!r}, {len(chosen.frames)} frames")
    lo, hi = 1.0, 0.0
    for camera in rig.unique_indices:
        path = chosen.tile_path(camera, frame)
        header = exr.probe(path)
        tick = time.perf_counter()
        bits = exr.load_half(path)
        half_ms = (time.perf_counter() - tick) * 1000
        tick = time.perf_counter()
        exr.load_levels(path)
        levels_ms = (time.perf_counter() - tick) * 1000
        same = np.array_equal(bits, ffmpeg_half(tools, path, (header.height, header.width)))
        linear = exr.half_to_linear(bits)
        lo, hi = min(lo, float(linear.min())), max(hi, float(linear.max()))
        print(
            f"camera {camera:2d}  : {header.width}x{header.height} {header.channels} "
            f"compression {header.compression}; half {half_ms:.0f} ms, levels {levels_ms:.0f} ms; "
            f"ffmpeg {'identical' if same else 'DIFFERS'}"
        )
    print(f"linear     : {lo:.4f} .. {hi:.4f}")
    indices = list(rig.unique_indices)
    for workers in (1, 4, 8):
        best = float("inf")
        for sample in chosen.frames[:3]:
            tick = time.perf_counter()
            io.load_tiles(chosen, sample, indices, workers=workers)
            best = min(best, time.perf_counter() - tick)
        print(f"frame      : {len(indices)} tiles, {workers} workers: {best:.2f} s (cached files)")
    return 0


# -- tiles ---------------------------------------------------------------------------------


def shared_frames(a: source.SourceSet, b: source.SourceSet, wanted: str | None) -> list[int]:
    frames = sorted(set(a.frames) & set(b.frames))
    if wanted:
        asked = [int(f) for f in wanted.split(",")]
        missing = sorted(set(asked) - set(frames))
        if missing:
            raise SystemExit(f"frames {missing} are not in both captures")
        frames = asked
    if not frames:
        raise SystemExit("the captures share no frame")
    return frames


def cmd_tiles(args: argparse.Namespace) -> int:
    a, b = capture(args.a), capture(args.b)
    rig = rig_for(a.camera_count)
    frames = shared_frames(a, b, args.frames)
    by = "EXR" if b.tile_format == "exr" else "A"
    print(f"A {a.root} ({a.tile_format})\nB {b.root} ({b.tile_format})")
    print(f"d = B - A, 8-bit levels, {len(frames)} frames, binned by the {by} level")
    total = np.zeros(4)  # samples, sum d, samples with |d| < 1, their sum d
    binned = np.zeros((len(BINS) - 1, 4))
    worst = 0.0
    for frame in frames:
        for camera in rig.unique_indices:
            left = levels(io.load_tile(a.tile_path(camera, frame))).ravel()
            right = levels(io.load_tile(b.tile_path(camera, frame))).ravel()
            d = (right - left).astype(np.float64)
            reference = right if b.tile_format == "exr" else left
            small = np.abs(d) < 1.0
            total += np.array([d.size, d.sum(), small.sum(), d[small].sum()], np.float64)
            worst = max(worst, float(np.abs(d).max()))
            cell = np.clip(np.digitize(reference, BINS) - 1, 0, len(BINS) - 2)
            for k in range(len(BINS) - 1):
                inside = cell == k
                both = inside & small
                counts = [inside.sum(), d[inside].sum(), both.sum(), d[both].sum()]
                binned[k] += np.array(counts, np.float64)
    n, s, ns, ss = total
    print(
        f"overall    : bias {s / n:+.4f}; |d| < 1 on {ns / n * 100:.2f}% of samples, "
        f"bias there {ss / max(ns, 1):+.4f}; max |d| {worst:.1f}"
    )
    for k in range(len(BINS) - 1):
        m, sm, mb, smb = binned[k]
        if m:
            print(
                f"level {BINS[k]:3d}-{BINS[k + 1]:<3d}: {m / n * 100:5.1f}% of samples, "
                f"bias {sm / m:+.4f}; |d| < 1 on {mb / m * 100:5.1f}%, "
                f"bias there {smb / max(mb, 1):+.4f}"
            )
    return 0


# -- stitch --------------------------------------------------------------------------------


def dithered(tiles: Mapping[int, Tile], seed: int) -> dict[int, U8]:
    """A synthetic PNG of float tiles: unbiased uniform dither, then floor -- the same
    picture at 8 bits with no systematic offset, so any bias left is the pipeline's."""
    rng = np.random.default_rng(seed)
    return {
        camera: np.asarray(
            np.clip(np.floor(levels(tile) + rng.random(tile.shape, dtype=np.float32)), 0, 255),
            dtype=np.uint8,
        )
        for camera, tile in tiles.items()
    }


def rounded(tiles: Mapping[int, Tile]) -> dict[int, U8]:
    """The io-only alternative to float tiles: the levels rounded to uint8 on read."""
    return {camera: np.asarray(np.rint(levels(tile)), np.uint8) for camera, tile in tiles.items()}


class Stitcher:
    def __init__(self, rig: Rig, tile: int, device: str, harmonised: bool) -> None:
        self.width = rig.native_width(tile)
        plan = WarpPlan.build(rig, self.width, self.width // 2, tile)
        self.plan: Warper = plan
        if device == "cuda":
            from vr_compose import warp_gpu

            self.plan = warp_gpu.GpuWarpPlan.from_plan(plan)
        self.harmoniser = harmonise.Harmoniser(rig, tile) if harmonised else None

    def run(
        self, tiles: Mapping[int, Tile], bit_depth: int
    ) -> tuple[io.Panorama, verify.Agreement]:
        grids = self.harmoniser.estimate(tiles).grids if self.harmoniser else None
        result = self.plan.apply(
            tiles, with_stats=True, threads=8, bit_depth=bit_depth, corrections=grids
        )
        return result.image, verify.agreement(result.stats)


def cmd_stitch(args: argparse.Namespace) -> int:
    a, b = capture(args.a), capture(args.b)
    rig = rig_for(a.camera_count)
    tile = a.tile_size
    if tile is None or tile != b.tile_size:
        raise SystemExit("the captures must share a square tile size")
    frames = shared_frames(a, b, args.frames)
    started = time.time()
    stitcher = Stitcher(rig, tile, args.device, not args.no_harmonise)
    print(f"A {a.root} ({a.tile_format})\nB {b.root} ({b.tile_format})")
    print(
        f"{stitcher.width}x{stitcher.width // 2}, catmullrom, harmonise "
        f"{'off' if args.no_harmonise else 'on'}, {args.device}; plan {time.time() - started:.1f} s"
    )
    indices = list(rig.unique_indices)
    for frame in frames:
        tiles_a = io.load_tiles(a, frame, indices, workers=8)
        tiles_b = io.load_tiles(b, frame, indices, workers=8)
        variants: list[tuple[str, Mapping[int, Tile]]] = [("A", tiles_a), ("B", tiles_b)]
        for name, loaded, chosen in (("A", tiles_a, a), ("B", tiles_b, b)):
            if chosen.tile_format == "exr":
                variants.append((f"{name} dithered", dithered(loaded, seed=frame)))
                variants.append((f"{name} rounded", rounded(loaded)))
        images: dict[tuple[str, int], io.Panorama] = {}
        for name, tiles in variants:
            for bit_depth in (8, 16):
                tick = time.time()
                image, agreement = stitcher.run(tiles, bit_depth)
                images[name, bit_depth] = image
                if bit_depth == 8:
                    print(
                        f"frame {frame} {name:10s}: {time.time() - tick:5.1f} s, metric A median "
                        f"{agreement.median:.3f} mean {agreement.mean:.3f} p95 {agreement.p95:.2f}"
                    )
        for bit_depth, scale in ((8, 1.0), (16, 257.0)):
            left, right = images["A", bit_depth], images["B", bit_depth]
            print(f"  {bit_depth:2d}-bit A -> B         : {diff_stats(left, right, scale)}")
            for name, _ in variants[2:]:
                base = name[0]
                left, right = images[name, bit_depth], images[base, bit_depth]
                print(
                    f"  {bit_depth:2d}-bit {name:10s} -> {base}: {diff_stats(left, right, scale)}"
                )
    return 0


# -- master --------------------------------------------------------------------------------


def ffmpeg_rgb(
    tools: encode.Tools, path: pathlib.Path, shape: tuple[int, int], pix_fmt: str
) -> U16:
    """ffmpeg's decode as 16-bit samples: `gbrpf16le` for an RGB EXR master (half bits
    untouched), `rgb48le` for a 16-bit PNG master."""
    command = [str(tools.ffmpeg), "-v", "error", "-i", str(path)]
    command += ["-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]
    raw = subprocess.run(command, capture_output=True, check=True).stdout
    samples = np.frombuffer(raw, np.uint16)
    if pix_fmt == "rgb48le":
        return samples.reshape(*shape, 3)
    g, b, r = samples.reshape(3, *shape)
    return np.stack([r, g, b], axis=-1)


def cmd_master(args: argparse.Namespace) -> int:
    tools = encode.find_tools()
    files = sorted(args.directory.glob("*.exr"))
    if not files:
        raise SystemExit(f"{args.directory}: no .exr master files")
    print(f"masters    : {args.directory}, {len(files)} file(s)")
    table = exr.srgb_levels_table()
    total_bytes = total_pixels = 0
    for path in files:
        header = exr.probe(path)
        shape = (header.height, header.width)
        pixels = header.width * header.height
        tick = time.perf_counter()
        bits = exr.load_half(path)
        seconds = time.perf_counter() - tick
        same = np.array_equal(bits, ffmpeg_rgb(tools, path, shape, "gbrpf16le"))
        size = path.stat().st_size
        total_bytes += size
        total_pixels += pixels
        print(
            f"{path.name}: {header.width}x{header.height} {header.channels} compression "
            f"{header.compression}; {size / 2**20:.1f} MiB, {size / pixels:.2f} bytes/pixel; "
            f"read {seconds:.2f} s; ffmpeg {'identical' if same else 'DIFFERS'}"
        )
        if args.png16 is not None:
            png = args.png16 / path.with_suffix(".png").name
            deep = ffmpeg_rgb(tools, png, shape, "rgb48le")
            same_bits = np.array_equal(bits, exr.half_from_levels16(deep))
            d = table[bits] - deep.astype(np.float32) / np.float32(257.0)
            print(
                f"  vs {png.name}: the PNG's codes through the table "
                f"{'identical' if same_bits else 'DIFFER'}; decoded back to levels: "
                f"bias {d.mean():+.5f}  mean|d| {np.abs(d).mean():.4f}  max {np.abs(d).max():.3f}"
            )
    print(f"bytes/pixel: {total_bytes / total_pixels:.2f} over {len(files)} file(s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="reader vs ffmpeg, decode timing")
    check.add_argument("capture", type=pathlib.Path)
    check.set_defaults(run=cmd_check)
    tiles = commands.add_parser("tiles", help="per-tile bias between two captures")
    tiles.add_argument("a", type=pathlib.Path)
    tiles.add_argument("b", type=pathlib.Path)
    tiles.add_argument("--frames", default=None, help="comma-separated; default: all shared")
    tiles.set_defaults(run=cmd_tiles)
    stitch = commands.add_parser("stitch", help="stitched panoramas of two captures compared")
    stitch.add_argument("a", type=pathlib.Path)
    stitch.add_argument("b", type=pathlib.Path)
    stitch.add_argument("--frames", default=None, help="comma-separated; default: all shared")
    stitch.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    stitch.add_argument("--no-harmonise", action="store_true")
    stitch.set_defaults(run=cmd_stitch)
    master = commands.add_parser("master", help="an EXR master directory vs ffmpeg, bytes/pixel")
    master.add_argument("directory", type=pathlib.Path)
    master.add_argument(
        "--png16", type=pathlib.Path, default=None,
        help="the 16-bit PNG master directory of the same run, to compare against",
    )  # fmt: skip
    master.set_defaults(run=cmd_master)
    args = parser.parse_args(argv)
    result: int = args.run(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
