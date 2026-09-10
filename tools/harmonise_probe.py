"""What does harmonisation do to a frame, and does it hold still over a sequence?

ROADMAP P9. For each frame: metric A before and after the correction, how much of the
output moved, and a side-by-side crop of the region that disagreed most. With `--temporal`
it also estimates the corrections for a run of consecutive frames and reports how much
the grids change frame to frame -- a correction that jumped would flicker in the headset.

    python tools/harmonise_probe.py --source "E:/0909C Data" --frames 100 900
    python tools/harmonise_probe.py --source E:/22 --frames 1656          # must change ~nothing
    python tools/harmonise_probe.py --source "E:/0909C Data" --frame 100 --temporal 30

Read-only on the source directory; images go to `--out-dir`.
"""

from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
from PIL import Image

from vr_compose import io, source, verify
from vr_compose.harmonise import Harmoniser
from vr_compose.rig import rig_for
from vr_compose.stitch import BandStats
from vr_compose.warp import WarpPlan

DEFAULT_ROOT = pathlib.Path("E:/22")


def crop_of_worst(before: np.ndarray, after: np.ndarray, stats: BandStats) -> Image.Image:
    """Left: before, right: after, around the 1200x600 window that disagreed most."""
    h, w = before.shape[:2]
    count = stats.count.reshape(h, w)
    mean = np.divide(
        stats.luma_sum, stats.count, out=np.zeros_like(stats.luma_sum), where=stats.count > 0
    )
    var = np.maximum(stats.luma_sq_sum / np.maximum(stats.count, 1) - mean**2, 0).reshape(h, w)
    std = np.where(count >= 2, np.sqrt(var), 0.0)
    block = 64
    coarse = (
        std[: h // block * block, : w // block * block]
        .reshape(h // block, block, w // block, block)
        .mean(axis=(1, 3))
    )
    cy, cx = np.unravel_index(int(np.argmax(coarse)), coarse.shape)
    ch, cw = min(600, h), min(1200, w)
    y = int(np.clip(cy * block - ch // 2, 0, h - ch))
    x = int(np.clip(cx * block - cw // 2, 0, w - cw))
    side = Image.new("RGB", (2 * cw + 10, ch), (255, 255, 255))
    side.paste(Image.fromarray(before[y : y + ch, x : x + cw]), (0, 0))
    side.paste(Image.fromarray(after[y : y + ch, x : x + cw]), (cw + 10, 0))
    return side


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=pathlib.Path, default=DEFAULT_ROOT)
    parser.add_argument("--stem", default=None)
    parser.add_argument("--frames", type=int, nargs="*", default=None)
    parser.add_argument("--frame", type=int, default=None, help="first frame of --temporal")
    parser.add_argument("--temporal", type=int, default=0, help="consecutive frames to track")
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--sampler", default="bilinear")
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("harmonise_probe_out"))
    args = parser.parse_args(argv)

    sets, searched = source.discover(args.source)
    if args.stem:
        sets = [s for s in sets if s.stem == args.stem]
    if len(sets) != 1:
        raise SystemExit(
            f"need exactly one source set (got {[s.stem for s in sets]}; searched {searched})"
        )
    chosen = sets[0]
    rig = rig_for(chosen.camera_count)
    tile = chosen.tile_size
    assert tile is not None
    indices = list(rig.unique_indices)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    harmoniser = Harmoniser(rig, tile)
    print(f"source : {chosen.root} stem {chosen.stem!r}, tile {tile}")
    print(
        f"scan   : {args.width}x{args.width // 2} {args.sampler}; estimation plan "
        f"{harmoniser.width}x{harmoniser.height} built in {harmoniser.build_seconds:.1f} s"
    )
    print()

    frames = args.frames or ([args.frame] if args.frame is not None else [chosen.frames[0]])
    plan = WarpPlan.build(rig, args.width, args.width // 2, tile, sampler=args.sampler)
    for frame in frames:
        tiles = io.load_tiles(chosen, frame, indices, workers=8)
        started = time.perf_counter()
        found = harmoniser.estimate(tiles)
        estimated = time.perf_counter() - started
        before = plan.apply(tiles, with_stats=True)
        after = plan.apply(tiles, with_stats=True, corrections=found.grids)
        a0, a1 = verify.agreement(before.stats), verify.agreement(after.stats)
        delta = np.abs(after.image.astype(np.int16) - before.image.astype(np.int16)).max(axis=2)
        print(f"frame {frame}:")
        print(
            f"  metric A   : median {a0.median:.2f} -> {a1.median:.2f}, "
            f"mean {a0.mean:.2f} -> {a1.mean:.2f}, "
            f">8 {a0.fraction_over_8:.2%} -> {a1.fraction_over_8:.2%}  "
            f"[{a0.verdict} -> {a1.verdict}]"
        )
        print(
            f"  correction : largest {found.max_abs:.1f} levels, estimate {estimated * 1e3:.0f} ms"
        )
        print(
            f"  output     : {(delta > 2).mean():.2%} of pixels moved >2 levels, "
            f"{(delta > 8).mean():.3%} >8, max {delta.max()}"
        )
        crop_of_worst(before.image, after.image, before.stats).save(
            args.out_dir / f"crop_{frame}.png"
        )
        Image.fromarray(after.image).save(args.out_dir / f"after_{frame}.png")
    print(f"  crops in {args.out_dir}")

    if args.temporal > 1:
        first = args.frame if args.frame is not None else frames[0]
        run = [f for f in chosen.frames if f >= first][: args.temporal]
        print(f"\ntemporal: frames {run[0]}..{run[-1]} ({len(run)})")
        previous = None
        jumps, maxes = [], []
        for frame in run:
            tiles = io.load_tiles(chosen, frame, indices, workers=8)
            found = harmoniser.estimate(tiles)
            stack = np.stack([found.grids[c] for c in sorted(found.grids)])
            if previous is not None:
                jumps.append(float(np.abs(stack - previous).max()))
            maxes.append(found.max_abs)
            previous = stack
        print(
            f"  largest correction per frame: median {np.median(maxes):.1f}, "
            f"max {max(maxes):.1f} levels"
        )
        print(
            f"  frame-to-frame change of the grids: median {np.median(jumps):.2f}, "
            f"p95 {np.percentile(jumps, 95):.2f}, max {max(jumps):.2f} levels"
        )
        print("  (a change under ~1 level a frame is invisible; a jump of several would flicker)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
