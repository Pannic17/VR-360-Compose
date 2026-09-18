"""How much of a small object survives the blend?

A blend that averages tiles which disagree does not usually look like a seam -- it looks
like a thin object that has gone half transparent, because the tile that drew it is
averaged with tiles that drew the background instead (AGENTS.md section 3, the Bay render).
Narrowing `--seam-band` is the lever; this says what each setting buys.
This measures exactly that: it finds small objects sitting in an otherwise flat part of
one camera's own view, then asks how much of each object's contrast against its
surroundings each way of blending keeps. 100% means the panorama kept the object exactly
as the camera drew it.

    python tools/blend_probe.py --source E:/0910B --frame 900 --pair 3 7
    python tools/blend_probe.py --source E:/0910B --frame 900 --bands 0.25
    python tools/blend_probe.py --source E:/0910B --frame 900 --pair 11 15 --width 7680

Objects are found in the *overlap* of the pair, since a direction only one tile can see
is not blended at all and has nothing to lose. Read-only on the source directory.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import numpy.typing as npt
from align_probe import warp_one  # a sibling in tools/, sharing the same warp

from vr_compose import io, source
from vr_compose.rig import Rig, rig_for
from vr_compose.stitch import stitch_frame
from vr_compose.warp import WarpPlan

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]
DEFAULT_ROOT = pathlib.Path("E:/0910B")
SURROUND = 12
"""Half-width of the neighbourhood an object has to sit alone in, in output pixels."""
OBJECT = 7
"""Half-width of the object itself."""
MIN_CONTRAST = 12.0
"""Levels an object must stand out by in the camera that drew it, or it is noise."""
FLAT = 16.0
"""How still the ring around an object has to be, in levels, for it to count as isolated.

Loose on purpose. The tight value the word "isolated" suggests (8) finds 17 objects in a
frame of the Bay render, and the percentiles then move with a single one; 16 finds about
90 and they settle. Measured both ways, the ordering of the blends does not change."""


def _boxes(values: F64, radius: int) -> F64:
    """Mean of every `2 * radius + 1` square, edges extended: a summed-area table."""
    padded = np.pad(values, radius + 1, mode="edge")
    table = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    n = 2 * radius + 1
    height, width = values.shape
    lower, upper = 0, n
    total = (
        table[upper : upper + height, upper : upper + width]
        - table[lower : lower + height, upper : upper + width]
        - table[upper : upper + height, lower : lower + width]
        + table[lower : lower + height, lower : lower + width]
    )
    return np.asarray(total / (n * n), dtype=np.float64)


def ring_of(luma: F64, y: int, x: int) -> F64:
    """The two columns just outside the object: what it is sitting on."""
    return np.concatenate(
        [
            luma[y - 2 * OBJECT : y + 2 * OBJECT, x - 2 * OBJECT],
            luma[y - 2 * OBJECT : y + 2 * OBJECT, x + 2 * OBJECT],
        ]
    )


def find_objects(luma: F64, shared: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Small departures from a locally flat surround: one coordinate each.

    Isolation is judged on the ring *around* the candidate rather than on a box centred
    on it -- a box centred on a fish contains the fish, so the fish itself makes its own
    neighbourhood look busy and no fish is ever found.
    """
    departure = np.abs(luma - _boxes(luma, SURROUND))
    candidates = (departure > MIN_CONTRAST) & shared
    height, width = luma.shape
    seen, found = set(), []
    for y, x in zip(*np.nonzero(candidates), strict=True):
        if not (2 * OBJECT < y < height - 2 * OBJECT and 2 * OBJECT < x < width - 2 * OBJECT):
            continue
        key = (y // (2 * OBJECT), x // (2 * OBJECT))
        if key in seen:
            continue
        seen.add(key)
        if ring_of(luma, int(y), int(x)).std() < FLAT:
            found.append((int(y), int(x)))
    return found


def strength(luma: F64, y: int, x: int, sign: float = 0.0) -> tuple[float, float]:
    """`(departure from the surrounding level, its sign)`, in 8-bit levels.

    The sign is settled once, on the camera that drew the object, and then imposed on
    every blend: a dark object measured by how bright its brightest pixel is would score
    near zero however well the blend kept it.
    """
    patch = luma[y - OBJECT : y + OBJECT + 1, x - OBJECT : x + OBJECT + 1]
    around = float(np.median(ring_of(luma, y, x)))
    up, down = float(patch.max()) - around, around - float(patch.min())
    if sign == 0.0:
        sign = 1.0 if up >= down else -1.0
    return (up if sign > 0 else down), sign


def to_luma(image: npt.NDArray[np.generic]) -> F64:
    values = np.asarray(image, dtype=np.float64)
    return 0.2126 * values[..., 0] + 0.7152 * values[..., 1] + 0.0722 * values[..., 2]


def ways_to_blend(
    tiles: dict[int, npt.NDArray[np.uint8]], rig: Rig, width: int, bands: list[float]
) -> dict[str, F64]:
    """The default, a harder feather, and one entry per seam band asked for."""
    size = next(iter(tiles.values())).shape[0]
    ways = {
        "default": to_luma(WarpPlan.build(rig, width, width // 2, size).apply(tiles).image),
        "feather 8": to_luma(stitch_frame(tiles, rig, width, feather_power=8.0).image),
    }
    for band in bands:
        plan = WarpPlan.build(rig, width, width // 2, size, seam_band=band)
        ways[f"seam band {band:g}"] = to_luma(plan.apply(tiles).image)
    return ways


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=pathlib.Path, default=DEFAULT_ROOT)
    parser.add_argument("--stem", default=None)
    parser.add_argument("--frame", type=int, default=None, help="default: the first frame")
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument(
        "--pair", type=int, nargs=2, default=(3, 7), metavar=("A", "B"),
        help="the two cameras whose overlap is searched for objects",
    )  # fmt: skip
    parser.add_argument(
        "--bands", default="0.5,0.25,0.1",
        help="seam bands to compare against the default blend",
    )  # fmt: skip
    args = parser.parse_args(argv)

    sets, searched = source.discover(args.source)
    if args.stem:
        sets = [s for s in sets if s.stem == args.stem]
    if len(sets) != 1:
        raise SystemExit(f"need exactly one source set (got {[s.stem for s in sets]}; {searched})")
    chosen = sets[0]
    rig = rig_for(chosen.camera_count)
    size = chosen.tile_size
    if size is None:
        raise SystemExit("source tiles are not square or not uniform")
    frame = args.frame if args.frame is not None else chosen.frames[0]
    a, b = args.pair

    tiles = io.load_tiles(chosen, frame, list(rig.unique_indices), workers=8)
    drawn, seen_a = warp_one(tiles, rig, a, size, args.width)
    _, seen_b = warp_one(tiles, rig, b, size, args.width)
    objects = find_objects(drawn, seen_a & seen_b)
    print(f"source : {chosen.root} stem {chosen.stem!r}, frame {frame}, {args.width} wide")
    print(f"objects: {len(objects)} in the C{a}/C{b} overlap, as C{a} drew them\n")
    if not objects:
        print("nothing isolated enough to measure -- try another frame or pair")
        return 1

    bands = [float(v) for v in args.bands.split(",") if v.strip()]
    blends = ways_to_blend(tiles, rig, args.width, bands)
    rows = []
    for y, x in objects:
        here, sign = strength(drawn, y, x)
        if here < MIN_CONTRAST:
            continue
        rows.append([here] + [strength(luma, y, x, sign)[0] for luma in blends.values()])
    table = np.array(rows)
    print(f"{'':24s}{'median kept':>12}{'p10':>7}{'worst':>7}{'under 70%':>11}")
    for i, label in enumerate(blends):
        kept = table[:, 1 + i] / table[:, 0]
        print(
            f"{label:24s}{np.median(kept) * 100:11.0f}%{np.percentile(kept, 10) * 100:6.0f}%"
            f"{kept.min() * 100:6.0f}%{np.mean(kept < 0.70) * 100:10.1f}%"
        )
    print(
        f"\n{len(table)} objects measured. 100% is what the one camera that drew it kept; "
        "anything well below means the blend averaged it with tiles that did not."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
