"""Temporal stability: does the pipeline add flicker the source does not have?

ROADMAP P4 item 5. Frames are stitched independently, so in principle nothing temporal
can be introduced -- but "in principle" is not evidence, and there are two plausible ways
to be wrong. A sampler that lands near a pixel boundary could flip which source pixel
dominates from frame to frame, and the extreme polar magnification (one source pixel over
~16 output pixels) would amplify any such flip.

The test is the one the criterion states: per-pixel temporal variance of the master must
not exceed that of the source tiles. Both are measured in levels squared over the same
consecutive frames, so the scene's own motion is common to them; anything the pipeline
adds shows up as the master exceeding the source.

The variance comparison turns out **not to be well posed at the pole**, and the probe
says so rather than reporting a failure. The two grids differ by about 16x there, so the
per-pixel variances are not comparable, and measurement bears that out: all three
samplers come out above 1.0 in the cap, with *nearest* the worst of them.

That last fact is what settles it. Nearest and bilinear are **convex** combinations of
source pixels -- every weight is non-negative and they sum to one, the feather blend
included -- so their output is bounded by their inputs and they cannot manufacture
variance at all. A ratio above 1 from those two is proof that the comparison is biased,
not that flicker was added.

So the probe also reports a **grid-free bound**: the largest frame-to-frame change any
output pixel undergoes, against the largest any source pixel undergoes. A convex resample
can never exceed its inputs, so master_max <= source_max must hold for nearest and
bilinear as a matter of arithmetic, and for catmullrom -- whose negative lobes give it a
gain above one in principle -- it is a real test rather than a tautology.

    python tools/temporal_probe.py
    python tools/temporal_probe.py --frames 10 --width 3840 --sampler bilinear
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import numpy.typing as npt

from vr_compose import io, source
from vr_compose.projection import camera_to_world, tile_rays
from vr_compose.rig import rig_for
from vr_compose.stitch import DEFAULT_SAMPLER
from vr_compose.warp import WarpPlan

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]

DEFAULT_ROOT = pathlib.Path("E:/22")
BANDS = (("cap >85", 85.0, 90.0), ("polar 70-85", 70.0, 85.0), ("mid 30-70", 30.0, 70.0),
         ("equator <30", 0.0, 30.0))  # fmt: skip


class TemporalStats:
    """Per-pixel variance plus the largest frame-to-frame change seen.

    Frames are 88 MB at 8K, so holding ten is 880 MB while two running float64
    accumulators plus the previous frame is about 700 MB -- and the accumulators let the
    frame count grow without changing the memory.
    """

    def __init__(self, size: int) -> None:
        self.total = np.zeros(size, np.float64)
        self.squares = np.zeros(size, np.float64)
        self.biggest_step = np.zeros(size, np.float64)
        self.previous: F64 | None = None
        self.count = 0

    def add(self, values: F64) -> None:
        self.total += values
        self.squares += values * values
        if self.previous is not None:
            np.maximum(self.biggest_step, np.abs(values - self.previous), out=self.biggest_step)
        self.previous = values
        self.count += 1

    def variance(self) -> F64:
        mean = self.total / self.count
        return np.asarray(np.maximum(self.squares / self.count - mean * mean, 0.0), np.float64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=DEFAULT_ROOT)
    parser.add_argument("--first", type=int, default=1656)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--width", type=int, default=7680, help="native width is fairest")
    parser.add_argument("--sampler", default=DEFAULT_SAMPLER)
    args = parser.parse_args(argv)

    found = [s for s in source.scan(args.root) if s.usable]
    if not found:
        print(f"no usable source set in {args.root}", file=sys.stderr)
        return 2
    chosen = found[0]
    rig = rig_for(chosen.camera_count)
    size = chosen.tile_size
    if size is None:
        print("source tiles are not square or not uniform", file=sys.stderr)
        return 2
    frames = [f for f in range(args.first, args.first + args.frames) if f in chosen.frames]
    if len(frames) < 3:
        print(f"need at least 3 consecutive frames from {args.first}", file=sys.stderr)
        return 2

    height = args.width // 2
    print(
        f"frames {frames[0]}..{frames[-1]} ({len(frames)}), master {args.width}x{height}, "
        f"sampler {args.sampler}\nbuilding the plan ...",
        flush=True,
    )
    plan = WarpPlan.build(rig, args.width, height, size, sampler=args.sampler)

    master = TemporalStats(args.width * height)
    tiles_by_camera = {index: TemporalStats(size * size) for index in rig.unique_indices}
    for frame in frames:
        loaded = io.load_tiles(chosen, frame, list(rig.unique_indices), workers=8)
        master.add(plan.apply(loaded).image.astype(np.float64).mean(axis=2).reshape(-1))
        for index, tile in loaded.items():
            tiles_by_camera[index].add(tile.astype(np.float64).mean(axis=2).reshape(-1))
        print(f"  frame {frame} done", flush=True)

    master_variance = master.variance()
    latitude = np.abs(np.repeat(90.0 - (np.arange(height) + 0.5) / height * 180.0, args.width))

    # the source's variance, grouped by the latitude each tile pixel lands on
    rays = tile_rays(size, rig.fov_deg, mirrored=rig.mirrored)
    source_by_band: dict[str, list[float]] = {name: [] for name, _, _ in BANDS}
    source_step_by_band: dict[str, list[float]] = {name: [] for name, _, _ in BANDS}
    for index in rig.unique_indices:
        view = rig.view_for(index)
        world = camera_to_world(view.yaw, view.elevation) @ rays
        tile_latitude = np.abs(np.degrees(np.arcsin(np.clip(world[2], -1.0, 1.0))))
        variance = tiles_by_camera[index].variance()
        steps = tiles_by_camera[index].biggest_step
        for name, lo, hi in BANDS:
            mask = (tile_latitude >= lo) & (tile_latitude < hi)
            if int(mask.sum()) > 1000:
                source_by_band[name].append(float(variance[mask].mean()))
                source_step_by_band[name].append(float(steps[mask].max()))

    print(
        f"\n  {'band':<13} {'master var':>11} {'source var':>11} {'ratio':>7}   "
        f"{'max step mine':>13} {'theirs':>7}  verdict"
    )
    worst_step = 0.0
    for name, lo, hi in BANDS:
        rows = (latitude >= lo) & (latitude < hi)
        if not rows.any() or not source_by_band[name]:
            continue
        mine = float(master_variance[rows].mean())
        theirs = float(np.mean(source_by_band[name]))
        ratio = mine / theirs if theirs > 0 else float("inf")
        my_step = float(master.biggest_step[rows].max())
        their_step = float(max(source_step_by_band[name]))
        step_ratio = my_step / their_step if their_step > 0 else float("inf")
        worst_step = max(worst_step, step_ratio)
        print(
            f"  {name:<13} {mine:11.3f} {theirs:11.3f} {ratio:7.3f}   "
            f"{my_step:13.1f} {their_step:7.1f}  "
            f"{'PASS' if step_ratio <= 1.0 else 'AMPLIFIED'}"
        )
    print(
        f"\nThe verdict column is the grid-free bound; worst max-step ratio {worst_step:.3f}."
        "\nA convex resample (nearest, bilinear) cannot exceed its inputs, so <= 1 there is"
        "\narithmetic rather than evidence; for catmullrom, whose negative lobes can amplify,"
        "\nit is a real test. The variance ratio is printed for the record but is not"
        "\ncomparable at the pole, where the two grids differ by about 16x -- see the"
        "\ndocstring at the top of this file."
    )
    return 0 if worst_step <= 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
