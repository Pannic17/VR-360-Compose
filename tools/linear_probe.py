"""P15 step 2: linear-light blending and the output encode, measured on an EXR source.

P4 turned linear-light blending down on 8-bit PNG tiles (0.017 to 0.040 levels,
`tools/linear_light_probe.py`). An EXR source changes two things and leaves one alone: the
decode is now free (the tiles *are* linear), the delivery MP4 and the PNG master still need
an sRGB encode, and the square law that made the blend a no-op -- overlapping tiles differ
by about half a level, and the gap between averaging in the two spaces grows with the
square of that -- never depended on the bit depth. So this measures again, on real EXR
tiles and the production geometry, instead of assuming either way.

    python tools/linear_probe.py variants <capture> [--frames 571,576] --out-dir DIR
                                     [--ab-frames 571-580] [--save-linear]
    python tools/linear_probe.py encode [--linear DIR/linear_blend_0571.npy] [--device cuda]

**variants** blends a frame on the one `WarpPlan` the product would use (catmullrom, feather
2, seam band 0.25, the rig's native width) in four arithmetics, so the geometry and the
weights are identical and only the space differs:

  V0  levels        step 1 as shipped: sRGB levels in, additive harmoniser, blend, quantise
  V1  blend-linear  V0 up to the blend, then each sample decoded and averaged in linear light
  V2  linear        linear in, linear interpolation and blend, no harmoniser; against V0 with
                    the harmoniser off, which isolates the interpolation
  V3  linear-gain   V2 plus the harmoniser as a *gain*: the same estimator run on log2 of the
                    linear tiles, so its additive field is a multiplicative one

V0 is checked byte for byte against `WarpPlan.apply` first, so the re-implementation here
is known to be the product's arithmetic and nothing else. The comparisons are on 16-bit
masters in levels (/257), split by overlap, by level, by local gradient and by latitude, with
heat maps and the worst 48x48 crop written to `--out-dir`; `--ab-frames` also writes 8-bit
8K PNGs of every variant for the headset.

**encode** is about what an adopted linear blend would still have to pay: the output encode,
and that it must be byte-identical on the CPU and the GPU (`powf` is not). Two table-driven
encodes are measured against the exact float64 curve over every float32 in [0, 1], over a
value-uniform sweep and over a real frame's blended values: float32 -> half (round to nearest
even on both sides) -> the reader's 65536-entry table; and a segment table indexed by the
float32 exponent and top eight mantissa bits with a linear interpolation on the rest. With
`--device cuda` both run as kernels (`--fmad=false`) and the bytes are compared.
Numbers in ROADMAP "P15" step 2 and AGENTS.md section 10.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import itertools
import pathlib
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from vr_compose import exr, harmonise, io, source
from vr_compose.rig import Rig, rig_for
from vr_compose.stitch import (
    _EPSILON,
    BandStats,
    bilinear_blend,
    cubic_blend,
    luma_bt709,
    quantise,
)
from vr_compose.warp import TileContributors, WarpPlan, upsample_grid

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]
U8 = npt.NDArray[np.uint8]
U16 = npt.NDArray[np.uint16]
U32 = npt.NDArray[np.uint32]
Bool = npt.NDArray[np.bool_]

VARIANTS = ("V0_levels", "V0_levels_noharm", "V1_blend_linear", "V2_linear", "V3_linear_gain")
LOG_SCALE = np.float32(32.0)
"""log2 of the linear tiles is scaled by this before the gain estimate, so the harmoniser's
clamp of 32 levels becomes a clamp of one stop (a gain between 0.5 and 2)."""
LOG_FLOOR = np.float32(2.0**-12)
"""Added to the linear value before the log: about level 0.8, so black stays finite and the
ratio of two near-black pixels does not dominate a cell."""
LEVEL_BINS = (0, 16, 64, 128, 200, 256)
GRADIENT_BINS = (("flat <2", 0.0, 2.0), ("textured 2-16", 2.0, 16.0), ("edge >=16", 16.0, 1e9))
DISAGREEMENT_BINS = (("agree <1", 0.0, 1.0), ("1-8", 1.0, 8.0), ("ghost >=8", 8.0, 1e9))
"""Standard deviation of the tiles' luma at a direction, in levels: the square law's own
variable, so the blend-space difference should grow with it."""
LATITUDE_BANDS = (("cap >85", 85.0, 90.1), ("polar 70-85", 70.0, 85.0),
                  ("mid 30-70", 30.0, 70.0), ("equator <30", 0.0, 30.0))  # fmt: skip
HEAT_GAIN = 32.0
"""Levels of difference that saturate the heat map."""
CROP = 48


# -- transfer functions --------------------------------------------------------------------


def decode_levels(levels: F32) -> F32:
    """sRGB levels (0..255, continuous) -> display linear, odd-symmetric past zero.

    Exact in float64. Overshoot from the cubic kernel is negative or above 255; mirroring the
    curve keeps that arithmetic symmetric with the product's, which clamps only at the end.
    """
    x = levels.astype(np.float64) / 255.0
    a = np.abs(x)
    linear = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    return np.asarray(np.sign(x) * linear, dtype=np.float32)


def encode_exact(linear: npt.NDArray[np.floating[Any]]) -> F64:
    """Display linear -> sRGB levels, exact in float64, odd-symmetric past zero."""
    x = np.asarray(linear, dtype=np.float64)
    a = np.abs(x)
    encoded = np.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1.0 / 2.4) - 0.055)
    return np.asarray(np.sign(x) * encoded * 255.0, dtype=np.float64)


def encode_levels(linear: F32) -> F32:
    """:func:`encode_exact` as float32 levels, the quantity the blend hands to `quantise`."""
    return np.asarray(encode_exact(linear), dtype=np.float32)


# -- the frame -----------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class Frame:
    """One frame's tiles in both spaces, flattened to ``(n*n, 3)`` for the plan's gathers."""

    levels: dict[int, F32]
    linear: dict[int, F32]
    square: dict[int, F32]
    """The levels as ``(n, n, 3)``, for `WarpPlan.apply` and the harmoniser."""


def load_frame(capture: source.SourceSet, frame: int, cameras: list[int]) -> Frame:
    table = exr.srgb_levels_table()
    levels, linear, square = {}, {}, {}
    for camera in cameras:
        bits = exr.load_half(capture.tile_path(camera, frame))
        square[camera] = np.asarray(table[bits], dtype=np.float32)
        levels[camera] = square[camera].reshape(-1, 3)
        linear[camera] = exr.half_to_linear(bits).reshape(-1, 3)
    return Frame(levels, linear, square)


def capture_set(path: pathlib.Path) -> source.SourceSet:
    found = [s for s in source.scan(path) if s.usable]
    if len(found) != 1:
        rejected = "\n".join(f"  {s.stem!r}: {s.problems}" for s in source.scan(path))
        raise SystemExit(f"{path}: expected one usable source set, found {len(found)}\n{rejected}")
    if found[0].tile_format != "exr":
        raise SystemExit(f"{path} is a {found[0].tile_format} capture; this probe reads EXR")
    return found[0]


def parse_frames(text: str | None, available: Sequence[int]) -> list[int]:
    if not text:
        return list(available[:1])
    wanted: list[int] = []
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            wanted.extend(range(int(lo), int(hi) + 1))
        else:
            wanted.append(int(part))
    missing = sorted(set(wanted) - set(available))
    if missing:
        raise SystemExit(f"frames {missing} are not in the capture")
    return wanted


# -- the blend, in a chosen space ---------------------------------------------------------


def gather(plan: WarpPlan, tile: TileContributors, flat: F32) -> F32:
    """The plan's sampler at this tile's entries: `WarpPlan.apply`'s arithmetic, any source."""
    base = tile.src_index
    if plan.sampler == "nearest":
        return np.asarray(flat[base], dtype=np.float32)
    step = plan.tile_size
    if plan.sampler == "bilinear":
        return bilinear_blend(
            (flat[base], flat[base + 1], flat[base + step], flat[base + step + 1]),
            tile.fx,
            tile.fy,
        )
    rows = tuple(
        (
            flat[base + k * step],
            flat[base + k * step + 1],
            flat[base + k * step + 2],
            flat[base + k * step + 3],
        )
        for k in range(4)
    )
    return cubic_blend(rows, tile.fx, tile.fy)


@dataclasses.dataclass(slots=True)
class Blend:
    """A variant's result: float32 levels per pixel, and where the seconds went."""

    levels: F32
    """``(pixels, 3)``, unquantised, in levels."""
    seconds: dict[str, float]

    def image(self, bit_depth: int, height: int, width: int) -> io.Panorama:
        return quantise(self.levels, bit_depth).reshape(height, width, 3)


def run_variant(
    plan: WarpPlan,
    frame: Frame,
    variant: str,
    additive: dict[int, F32] | None,
    gain: dict[int, F32] | None,
    threads: int = 8,
) -> Blend:
    """One variant's blend. Tiles are gathered concurrently (numpy releases the GIL in the
    gathers) and accumulated in plan order, so the float32 sums add up as the product's do."""
    pixels = plan.width * plan.height
    in_linear = variant in ("V2_linear", "V3_linear_gain")
    tiles = [tile for tile in plan.tiles if tile.out_index.size]

    def sample(tile: TileContributors) -> F32:
        sampled = gather(
            plan, tile, frame.linear[tile.camera] if in_linear else frame.levels[tile.camera]
        )
        if variant in ("V0_levels", "V1_blend_linear") and additive is not None:
            sampled = sampled - upsample_grid(
                additive[tile.camera], tile.out_index, plan.width, plan.height
            )
        if variant == "V3_linear_gain" and gain is not None:
            field = upsample_grid(gain[tile.camera], tile.out_index, plan.width, plan.height)
            sampled = sampled * np.exp2(-field / LOG_SCALE)
        return sampled

    seconds = {"gather": 0.0, "decode": 0.0, "encode": 0.0}
    tick = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
        samples = list(pool.map(sample, tiles))
    seconds["gather"] = time.perf_counter() - tick
    colour = np.zeros((pixels, 3), np.float32)
    weight = np.zeros(pixels, np.float32)
    for tile, sampled in zip(tiles, samples, strict=True):
        if variant == "V1_blend_linear":
            tick = time.perf_counter()
            sampled = decode_levels(sampled)
            seconds["decode"] += time.perf_counter() - tick
        colour[tile.out_index] += sampled * tile.weight[:, None]
        weight[tile.out_index] += tile.weight
    blended = np.asarray(colour / np.maximum(weight, _EPSILON)[:, None], dtype=np.float32)
    if variant in ("V1_blend_linear", "V2_linear", "V3_linear_gain"):
        tick = time.perf_counter()
        blended = encode_levels(blended)
        seconds["encode"] = time.perf_counter() - tick
    return Blend(blended, seconds)


def gain_grids(harmoniser: harmonise.Harmoniser, frame: Frame, size: int) -> dict[int, F32]:
    """The harmoniser's estimator on log2 of the linear tiles: an additive field there is a
    gain on the samples. `LOG_SCALE` makes its 32-level clamp a one-stop clamp."""
    logs = {
        camera: np.asarray(
            np.log2(np.maximum(flat, 0.0) + LOG_FLOOR) * LOG_SCALE, dtype=np.float32
        ).reshape(size, size, 3)
        for camera, flat in frame.linear.items()
    }
    return harmoniser.estimate(logs).grids


# -- comparing two masters -----------------------------------------------------------------


def latitude_rows(height: int) -> F64:
    return np.abs(90.0 - (np.arange(height) + 0.5) / height * 180.0)


def gradient_levels(luma: F32, height: int, width: int) -> F32:
    """Largest step to the right or downward neighbour, in levels, per pixel."""
    grid = luma.reshape(height, width)
    right = np.zeros_like(grid)
    down = np.zeros_like(grid)
    right[:, :-1] = np.abs(grid[:, 1:] - grid[:, :-1])
    down[:-1, :] = np.abs(grid[1:, :] - grid[:-1, :])
    return np.asarray(np.maximum(right, down).reshape(-1), dtype=np.float32)


@dataclasses.dataclass(slots=True)
class Splits:
    """Pixel masks a difference is broken down by. Built once per frame from V0."""

    named: list[tuple[str, str, Bool]]
    """``(group, label, mask over pixels)``."""

    @classmethod
    def build(cls, plan: WarpPlan, reference: F32, stats: BandStats) -> Splits:
        pixels = plan.width * plan.height
        # every direction of this rig lies inside two or more frusta, so "overlap" is the whole
        # sphere; what decides whether a blend happens is how many tiles keep weight after the
        # seam band
        weighted = np.bincount(
            np.concatenate([t.out_index[t.weight > 0] for t in plan.tiles]), minlength=pixels
        )
        n = np.maximum(stats.count, 1.0)
        mean = stats.luma_sum / n
        spread = np.sqrt(np.maximum(stats.luma_sq_sum / n - mean * mean, 0.0))
        luma = luma_bt709(reference)
        level = np.clip(luma, 0.0, 255.0)
        gradient = gradient_levels(luma, plan.height, plan.width)
        latitude = np.repeat(latitude_rows(plan.height), plan.width)
        named: list[tuple[str, str, Bool]] = [
            ("weighted", "one tile", weighted < 2),
            ("weighted", "two or more", weighted >= 2),
        ]
        for label, s_lo, s_hi in DISAGREEMENT_BINS:
            named.append(("tiles std", label, (spread >= s_lo) & (spread < s_hi)))
        for lo, hi in itertools.pairwise(LEVEL_BINS):
            named.append(("level", f"{lo}-{hi}", (level >= lo) & (level < hi)))
        for label, g_lo, g_hi in GRADIENT_BINS:
            named.append(("gradient", label, (gradient >= g_lo) & (gradient < g_hi)))
        for label, l_lo, l_hi in LATITUDE_BANDS:
            named.append(("latitude", label, (latitude >= l_lo) & (latitude < l_hi)))
        return cls(named)


def compare(
    name: str, a: Blend, b: Blend, splits: Splits, plan: WarpPlan, report_dir: pathlib.Path
) -> dict[str, float]:
    """Print the difference `b - a` of the 16-bit masters in levels, and save its pictures."""
    height, width = plan.height, plan.width
    a16 = quantise(a.levels, 16).astype(np.float64)
    b16 = quantise(b.levels, 16).astype(np.float64)
    d = (b16 - a16) / 257.0
    ad = np.abs(d)
    worst = np.max(ad, axis=1)
    a8 = quantise(a.levels, 8)
    b8 = quantise(b.levels, 8)
    changed = float(np.mean(np.any(a8 != b8, axis=1)) * 100)
    p99, p999 = np.percentile(ad, 99), np.percentile(ad, 99.9)
    print(f"  {name}")
    print(
        f"    all          bias {d.mean():+.4f}  mean|d| {ad.mean():.4f}  p99 {p99:.3f}  "
        f"p99.9 {p999:.3f}  max {ad.max():.2f}   8-bit pixels changed {changed:.3f}%"
    )
    for group, label, mask in splits.named:
        if not mask.any():
            continue
        part = d[mask]
        print(
            f"    {group:<10}{label:<13} {float(mask.mean()) * 100:5.1f}%  "
            f"bias {part.mean():+.4f}  mean|d| {np.abs(part).mean():.4f}  "
            f"max {np.abs(part).max():.2f}"
        )
    report_dir.mkdir(parents=True, exist_ok=True)
    heat = np.clip(worst.reshape(height, width) / HEAT_GAIN * 255.0, 0.0, 255.0)
    small = heat.reshape(height // 4, 4, width // 4, 4).mean(axis=(1, 3))
    io.write_png(
        report_dir / f"{name}_heat.png", np.repeat(small[..., None], 3, axis=2).astype(np.uint8)
    )
    at = int(np.argmax(worst))
    row, col = divmod(at, width)
    r0 = int(np.clip(row - CROP // 2, 0, height - CROP))
    c0 = int(np.clip(col - CROP // 2, 0, width - CROP))
    left = a8.reshape(height, width, 3)[r0 : r0 + CROP, c0 : c0 + CROP]
    right = b8.reshape(height, width, 3)[r0 : r0 + CROP, c0 : c0 + CROP]
    gap = np.full((CROP, 2, 3), 255, np.uint8)
    pair = np.concatenate([left, gap, right], axis=1).repeat(8, axis=0).repeat(8, axis=1)
    io.write_png(report_dir / f"{name}_worst_{row}_{col}.png", pair)
    print(f"    worst at row {row} col {col} (lat {latitude_rows(height)[row]:.1f} deg)")
    return {"bias": float(d.mean()), "mean": float(ad.mean()), "max": float(ad.max())}


# -- variants ------------------------------------------------------------------------------


def cmd_variants(args: argparse.Namespace) -> int:
    capture = capture_set(args.capture)
    rig: Rig = rig_for(capture.camera_count)
    size = capture.tile_size
    if size is None:
        raise SystemExit("tiles are not square or not uniform")
    frames = parse_frames(args.frames, capture.frames)
    ab_frames = parse_frames(args.ab_frames, capture.frames) if args.ab_frames else []
    width = args.width or rig.native_width(size)
    height = width // 2
    out_dir: pathlib.Path = args.out_dir
    report_dir = out_dir / "P15s2_reports"
    cameras = list(rig.unique_indices)

    tick = time.perf_counter()
    plan = WarpPlan.build(rig, width, height, size)
    harmoniser = harmonise.Harmoniser(rig, size)
    print(
        f"{capture.root}\n{width}x{height} {plan.sampler}, feather {plan.feather_power:g}, "
        f"seam band {plan.seam_band:g}; plan + harmoniser {time.perf_counter() - tick:.1f} s\n"
    )

    for frame_number in frames:
        tick = time.perf_counter()
        frame = load_frame(capture, frame_number, cameras)
        print(f"frame {frame_number}: tiles read in {time.perf_counter() - tick:.1f} s")
        tick = time.perf_counter()
        additive = harmoniser.estimate(frame.square)
        t_add = time.perf_counter() - tick
        tick = time.perf_counter()
        gain = gain_grids(harmoniser, frame, size)
        t_gain = time.perf_counter() - tick
        gain_max = max(float(np.abs(g).max()) for g in gain.values()) / float(LOG_SCALE)
        print(
            f"  harmoniser additive: largest {additive.max_abs:.1f} levels ({t_add:.2f} s); "
            f"as gain: largest {gain_max:.2f} stops ({t_gain:.2f} s)"
        )

        blends: dict[str, Blend] = {}
        for variant in VARIANTS:
            tick = time.perf_counter()
            blends[variant] = run_variant(
                plan,
                frame,
                variant,
                additive.grids if variant != "V0_levels_noharm" else None,
                gain,
            )
            total = time.perf_counter() - tick
            parts = ", ".join(f"{k} {v:.1f}" for k, v in blends[variant].seconds.items() if v)
            print(f"  {variant:<17} {total:6.1f} s  ({parts})")

        tick = time.perf_counter()
        shipped = plan.apply(
            frame.square, with_stats=True, threads=8, bit_depth=16, corrections=additive.grids
        )
        same = np.array_equal(shipped.image, blends["V0_levels"].image(16, height, width))
        print(
            f"  V0 vs WarpPlan.apply (16-bit): {'identical' if same else 'DIFFERS'} "
            f"({time.perf_counter() - tick:.1f} s)"
        )
        if not same:
            raise SystemExit("the probe's V0 is not the product's arithmetic; stop here")

        splits = Splits.build(plan, blends["V0_levels"].levels, shipped.stats)
        print("  differences on the 16-bit masters, levels (b - a):")
        tag = f"{frame_number:04d}"
        compare(
            f"{tag}_V1_vs_V0",
            blends["V0_levels"],
            blends["V1_blend_linear"],
            splits,
            plan,
            report_dir,
        )
        compare(
            f"{tag}_V2_vs_V0noharm",
            blends["V0_levels_noharm"],
            blends["V2_linear"],
            splits,
            plan,
            report_dir,
        )
        compare(
            f"{tag}_V3_vs_V0",
            blends["V0_levels"],
            blends["V3_linear_gain"],
            splits,
            plan,
            report_dir,
        )
        compare(
            f"{tag}_V3_vs_V2",
            blends["V2_linear"],
            blends["V3_linear_gain"],
            splits,
            plan,
            report_dir,
        )
        compare(
            f"{tag}_harm_vs_noharm",
            blends["V0_levels_noharm"],
            blends["V0_levels"],
            splits,
            plan,
            report_dir,
        )
        if args.save_linear:
            report_dir.mkdir(parents=True, exist_ok=True)
            linear_blend = decode_levels(blends["V2_linear"].levels)
            np.save(report_dir / f"linear_blend_{tag}.npy", linear_blend)
        print()

    if ab_frames:
        print(f"A/B PNGs for frames {ab_frames[0]}..{ab_frames[-1]} into {out_dir}")
        for frame_number in ab_frames:
            frame = load_frame(capture, frame_number, cameras)
            additive = harmoniser.estimate(frame.square)
            gain = gain_grids(harmoniser, frame, size)
            for variant in VARIANTS:
                blend = run_variant(
                    plan,
                    frame,
                    variant,
                    additive.grids if variant != "V0_levels_noharm" else None,
                    gain,
                )
                path = out_dir / f"P15s2_{variant}" / f"{capture.stem}_S_{frame_number:04d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                io.write_png(
                    path, np.asarray(blend.image(8, height, width), np.uint8), compress_level=1
                )
            print(f"  frame {frame_number} written")
    return 0


# -- encode --------------------------------------------------------------------------------

SEGMENT_SHIFT = 15
SEGMENTS = (0x3F800000 >> SEGMENT_SHIFT) + 1
"""Entries up to and including the one for 1.0; one more is stored so the last lerp reads
a neighbour."""


def segment_table() -> F32:
    """Exact levels at the start of every segment: float32 bits ``i << 15``, i in 0..SEGMENTS."""
    bits = (np.arange(SEGMENTS + 1, dtype=np.uint64) << SEGMENT_SHIFT).astype(np.uint32)
    starts = np.minimum(bits.view(np.float32).astype(np.float64), 1.0)
    return np.asarray(encode_exact(starts), dtype=np.float32)


FLOOR = np.float32(2.0**-64)
"""Inputs below this encode as 0 on both sides. cupy compiles every kernel with `-ftz=true`
(appended after the caller's options), so the GPU flushes denormal operands and results;
above this floor no operand or intermediate of either encode is denormal, and below it the
exact level is under 2e-16."""


def clamp_unit(linear: F32) -> F32:
    """What both encodes see: NaN to 0, then 0..1, then the floor -- which also turns -0.0
    into +0.0 (a clamp keeps the sign bit, and the segment index reads it)."""
    clamped = np.clip(np.nan_to_num(linear, nan=0.0), 0.0, 1.0).astype(np.float32)
    return np.asarray(np.where(clamped < FLOOR, np.float32(0.0), clamped), dtype=np.float32)


def encode_half_table(linear: F32, table: F32) -> F32:
    """float32 -> half (round to nearest even) -> the reader's table."""
    codes = clamp_unit(linear).astype(np.float16).view(np.uint16)
    return np.asarray(table[codes], dtype=np.float32)


def encode_segment_table(linear: F32, table: F32) -> F32:
    """Exponent and top eight mantissa bits pick the segment, the other fifteen interpolate.

    `y0 + t * (y1 - y0)` in float32, each operation rounded on its own; a CUDA kernel built
    with `--fmad=false` does the same three roundings.
    """
    bits = clamp_unit(linear).view(np.uint32)
    index = bits >> SEGMENT_SHIFT
    t = (bits & np.uint32((1 << SEGMENT_SHIFT) - 1)).astype(np.float32) * np.float32(
        1.0 / (1 << SEGMENT_SHIFT)
    )
    y0 = table[index]
    y1 = table[index + 1]
    diff = y1 - y0
    prod = t * diff
    return np.asarray(y0 + prod, dtype=np.float32)


def encode_powf(linear: F32) -> F32:
    """The direct float32 encode, for the time it costs; not byte-identical anywhere."""
    a = clamp_unit(linear)
    toe = a * np.float32(12.92)
    curve = np.float32(1.055) * np.power(a, np.float32(1.0 / 2.4)) - np.float32(0.055)
    return np.asarray(
        np.where(a <= np.float32(0.0031308), toe, curve) * np.float32(255.0), np.float32
    )


_KERNELS = r"""
#include <cuda_fp16.h>
extern "C" __global__ void encode_half(
    const float* __restrict__ linear, const float* __restrict__ table, long long n, float* out)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = fminf(fmaxf(linear[i], 0.0f), 1.0f);          // NaN -> 0, then 0..1
    if (v < 5.42101086242752217e-20f) v = 0.0f;              // FLOOR = 2^-64; -0 -> +0
    unsigned short code = __half_as_ushort(__float2half_rn(v));
    out[i] = table[code];
}
extern "C" __global__ void encode_segment(
    const float* __restrict__ linear, const float* __restrict__ table, long long n, float* out)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = fminf(fmaxf(linear[i], 0.0f), 1.0f);
    if (v < 5.42101086242752217e-20f) v = 0.0f;
    unsigned int bits = __float_as_uint(v);
    unsigned int index = bits >> 15;
    float t = (float)(bits & 0x7FFFu) * (1.0f / 32768.0f);
    float y0 = table[index];
    float y1 = table[index + 1];
    float diff = y1 - y0;
    float prod = t * diff;
    out[i] = y0 + prod;
}
"""


class GpuEncoders:
    def __init__(self, half: F32, segment: F32) -> None:
        import cupy  # type: ignore[import-untyped,unused-ignore]

        self.cp = cupy
        module = cupy.RawModule(code=_KERNELS, options=("--fmad=false",))
        self.half_kernel = module.get_function("encode_half")
        self.segment_kernel = module.get_function("encode_segment")
        self.half_table = cupy.asarray(half)
        self.segment_table = cupy.asarray(segment)

    def run(self, kernel: Any, table: Any, linear: F32) -> F32:
        cp = self.cp
        device = cp.asarray(linear)
        out = cp.empty_like(device)
        n = int(device.size)
        block = 256
        kernel(((n + block - 1) // block,), (block,), (device, table, np.int64(n), out))
        return np.asarray(cp.asnumpy(out), dtype=np.float32)

    def half(self, linear: F32) -> F32:
        return self.run(self.half_kernel, self.half_table, linear)

    def segment(self, linear: F32) -> F32:
        return self.run(self.segment_kernel, self.segment_table, linear)


@dataclasses.dataclass(slots=True)
class Tally:
    """Errors of one encode against the exact curve, accumulated over chunks."""

    name: str
    count: int = 0
    abs_sum: float = 0.0
    abs_max: float = 0.0
    at: float = 0.0
    """The input where `abs_max` was seen."""
    off8: int = 0
    off16: int = 0
    gpu_mismatch: int = 0
    seconds: float = 0.0

    def add(self, linear: F32, got: F32, exact: F64, gpu: F32 | None) -> None:
        error = np.abs(got.astype(np.float64) - exact)
        self.count += int(error.size)
        self.abs_sum += float(error.sum())
        k = int(np.argmax(error))
        if error[k] > self.abs_max:
            self.abs_max, self.at = float(error[k]), float(linear[k])
        self.off8 += int(np.count_nonzero(np.rint(got) != np.rint(exact)))
        self.off16 += int(
            np.count_nonzero(np.rint(got * np.float32(257.0)) != np.rint(exact * 257.0))
        )
        if gpu is not None:
            self.gpu_mismatch += int(np.count_nonzero(got.view(np.uint32) != gpu.view(np.uint32)))

    def line(self, gpu: bool) -> str:
        text = (
            f"    {self.name:<12} mean|e| {self.abs_sum / max(self.count, 1):.2e}  "
            f"max|e| {self.abs_max:.2e} levels at {self.at:.6g}   "
            f"8-bit off {self.off8 / max(self.count, 1) * 100:.4f}%   "
            f"16-bit off {self.off16 / max(self.count, 1) * 100:.3f}%"
        )
        if gpu and self.name != "powf":
            text += f"   GPU bytes differ: {self.gpu_mismatch}"
        return text


def evaluate(
    label: str,
    chunks: Callable[[], Any],
    encoders: dict[str, Callable[[F32], F32]],
    gpu: GpuEncoders | None,
) -> None:
    tallies = {name: Tally(name) for name in encoders}
    for linear in chunks():
        exact = encode_exact(clamp_unit(linear))
        for name, encoder in encoders.items():
            tick = time.perf_counter()
            got = encoder(linear)
            tallies[name].seconds += time.perf_counter() - tick
            device = None
            if gpu is not None and name in ("half", "segment"):
                device = gpu.half(linear) if name == "half" else gpu.segment(linear)
            tallies[name].add(linear, got, exact, device)
    print(f"  {label}: {next(iter(tallies.values())).count:,} values")
    for tally in tallies.values():
        print(tally.line(gpu is not None))


def cmd_encode(args: argparse.Namespace) -> int:
    half = exr.srgb_levels_table()
    segment = segment_table()
    gpu = GpuEncoders(half, segment) if args.device == "cuda" else None
    encoders: dict[str, Callable[[F32], F32]] = {
        "half": lambda x: encode_half_table(x, half),
        "segment": lambda x: encode_segment_table(x, segment),
        "powf": encode_powf,
    }
    print(
        f"segment table: {SEGMENTS + 1} entries ({(SEGMENTS + 1) * 4 / 1024:.0f} KiB); "
        "half table 65536 entries\n"
    )

    def half_codes() -> Any:
        yield np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)

    def value_uniform() -> Any:
        yield np.linspace(0.0, 1.0, 1 << 24, dtype=np.float64).astype(np.float32)

    def every_float() -> Any:
        last = 0x3F800000
        step = 1 << 24
        for lo in range(0, last + 1, step):
            hi = min(lo + step, last + 1)
            yield np.arange(lo, hi, dtype=np.uint32).view(np.float32)

    evaluate("the 65536 half codes (clamped to 0..1)", half_codes, encoders, gpu)
    evaluate("2^24 values uniform in [0, 1]", value_uniform, encoders, gpu)
    if not args.quick:
        evaluate("every float32 in [0, 1]", every_float, encoders, gpu)
    if args.linear is not None:
        blend = np.load(args.linear).reshape(-1)
        print(f"\n  real frame {args.linear.name}: {blend.size:,} values, timing is one 8K frame")
        tallies = {name: Tally(name) for name in encoders}
        exact = encode_exact(clamp_unit(blend))
        for name, encoder in encoders.items():
            best = float("inf")
            got = encoder(blend)
            for _ in range(3):
                tick = time.perf_counter()
                got = encoder(blend)
                best = min(best, time.perf_counter() - tick)
            device = None
            if gpu is not None and name != "powf":
                device = gpu.half(blend) if name == "half" else gpu.segment(blend)
            tallies[name].add(blend, got, exact, device)
            tallies[name].seconds = best
        for tally in tallies.values():
            print(tally.line(gpu is not None) + f"   CPU {tally.seconds:.2f} s")
    print(
        "\n  Read: a 16-bit master steps every 0.0039 levels, 8-bit delivery every 1; an encode\n"
        "  whose error stays under half of that never moves a rounded output."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    variants = commands.add_parser("variants", help="four blend arithmetics on one plan")
    variants.add_argument("capture", type=pathlib.Path)
    variants.add_argument("--frames", default=None, help="comma list or a-b; default: first")
    variants.add_argument("--ab-frames", default=None, help="also write 8-bit PNGs of these")
    variants.add_argument("--out-dir", type=pathlib.Path, required=True)
    variants.add_argument("--width", type=int, default=None, help="default: native")
    variants.add_argument("--save-linear", action="store_true", help="V2's linear blend as .npy")
    variants.set_defaults(run=cmd_variants)
    encode = commands.add_parser("encode", help="table encodes against the exact curve")
    encode.add_argument("--linear", type=pathlib.Path, default=None, help="a saved linear blend")
    encode.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    encode.add_argument("--quick", action="store_true", help="skip the exhaustive float32 sweep")
    encode.set_defaults(run=cmd_encode)
    args = parser.parse_args(argv)
    result: int = args.run(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
