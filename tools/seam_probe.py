"""Seams and the feather exponent: is a tile boundary visible, and what does it cost?

Two P4 criteria live here.

**Item 4, the seam check.** "No visible seam" is made falsifiable by sampling equally
spaced positions along the great circles where tiles meet, and asking whether the image
gradient across the boundary is unusual compared with the gradient along it. A seam is a
discontinuity in one direction only, so the ratio of the two is the signal. A ratio near
1 means the boundary is indistinguishable from ordinary image content; a step edge would
push it well above 1.

**Item 2's premise.** The plan asserted that averaging misaligned tiles is "actively
blurring" and that the poles want a joint reconstruction instead. The feather exponent
spans that whole family -- raise it and the blend slides from averaging everything toward
taking only the most central tile -- so sweeping it tests the premise directly, using the
round-trip fidelity measure rather than the agreement metric, which prefers blur.

    python tools/seam_probe.py
    python tools/seam_probe.py --powers 2 8 --width 3840
"""

from __future__ import annotations

import argparse
import itertools
import pathlib
import sys

import numpy as np
import numpy.typing as npt

from vr_compose import io, source, verify
from vr_compose.rig import Rig, rig_for
from vr_compose.stitch import stitch_frame

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]
Panorama = npt.NDArray[np.uint8] | npt.NDArray[np.uint16]

DEFAULT_ROOT = pathlib.Path("E:/22")
SAMPLES = 20
"""Positions per boundary, as ROADMAP P4 asks for."""
SPAN = 12
"""Pixels either side of a boundary to measure over."""


def boundary_directions(rig: Rig, samples: int) -> list[tuple[float, float, str]]:
    """`(longitude, latitude, label)` on the seams between neighbouring tiles.

    The vertical seams sit halfway between adjacent yaws; the horizontal ones halfway
    between adjacent elevations. Those are the meridians and parallels where one tile's
    feather has fallen to zero and its neighbour's has not, i.e. where a seam would be.
    """
    yaws = sorted({view.yaw % 360.0 for view in rig.unique_views.values()})
    elevations = sorted({view.elevation for view in rig.unique_views.values()})
    places: list[tuple[float, float, str]] = []
    for a, b in itertools.pairwise([*yaws, yaws[0] + 360.0]):
        seam_yaw = (a + b) / 2.0
        for i in range(samples):
            latitude = -60.0 + 120.0 * (i + 0.5) / samples
            places.append((seam_yaw, latitude, f"yaw {seam_yaw:.0f}"))
    for a, b in itertools.pairwise(elevations):
        seam_lat = (a + b) / 2.0
        for i in range(samples):
            longitude = 360.0 * (i + 0.5) / samples
            places.append((longitude, seam_lat, f"lat {seam_lat:+.0f}"))
    return places


def gradient_ratio(
    image: Panorama, longitude: float, latitude: float, vertical_seam: bool
) -> float:
    """Mean |gradient| across the boundary over mean |gradient| along it.

    Both are measured on the same little patch, so scene content cancels: a smooth patch
    has small gradients in both directions and a busy one has large gradients in both.
    Only a discontinuity in one direction moves the ratio.
    """
    height, width = image.shape[:2]
    column = int((longitude / 360.0 % 1.0) * width)
    row = int(np.clip((0.5 - latitude / 180.0) * height, SPAN + 1, height - SPAN - 2))
    columns = np.arange(column - SPAN, column + SPAN + 1) % width
    rows = np.arange(row - SPAN, row + SPAN + 1)
    patch = image[np.ix_(rows, columns)].astype(np.float32).mean(axis=2)
    across = np.abs(np.diff(patch, axis=1 if vertical_seam else 0))
    along = np.abs(np.diff(patch, axis=0 if vertical_seam else 1))
    denominator = float(along.mean())
    if denominator < 1e-3:
        return 1.0  # a flat patch: no seam can hide here, and the ratio is meaningless
    return float(across.mean()) / denominator


def seam_report(image: Panorama, rig: Rig, samples: int) -> dict[str, float]:
    ratios: list[float] = []
    for longitude, latitude, label in boundary_directions(rig, samples):
        ratios.append(gradient_ratio(image, longitude, latitude, label.startswith("yaw")))
    array = np.asarray(ratios, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
        "over_1_5": float((array > 1.5).mean()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=DEFAULT_ROOT)
    parser.add_argument("--frame", type=int, default=1656)
    parser.add_argument("--width", type=int, default=7680)
    parser.add_argument("--sampler", default=None, help="default: the product default")
    parser.add_argument(
        "--powers",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
        help="feather exponents to sweep; 2 is the default blend",
    )
    parser.add_argument("--samples", type=int, default=SAMPLES)
    args = parser.parse_args(argv)

    found = [s for s in source.scan(args.root) if s.usable]
    if not found:
        print(f"no usable source set in {args.root}", file=sys.stderr)
        return 2
    chosen = found[0]
    rig = rig_for(chosen.camera_count)
    tiles = io.load_tiles(chosen, args.frame, list(rig.unique_indices), workers=8)
    kwargs = {"sampler": args.sampler} if args.sampler else {}

    places = boundary_directions(rig, args.samples)
    print(
        f"frame {args.frame}, {args.width}x{args.width // 2}, "
        f"{len(places)} boundary positions ({args.samples} per seam)\n"
    )
    print(
        f"  {'feather':>8} {'seam median':>12} {'p95':>7} {'max':>7} {'>1.5':>7}"
        f" {'metric A':>9} {'wrap':>6}"
    )
    for power in args.powers:
        result = stitch_frame(tiles, rig, args.width, feather_power=power, **kwargs)
        seams = seam_report(result.image, rig, args.samples)
        agreement = verify.agreement(result.stats)
        print(
            f"  {power:8.1f} {seams['median']:12.3f} {seams['p95']:7.3f} {seams['max']:7.3f}"
            f" {seams['over_1_5']:6.1%} {agreement.median:9.3f}"
            f" {verify.wrap_seam_error(result.image):6.3f}"
        )
    print(
        "\nSeam ratio is |gradient across the boundary| / |gradient along it|, so 1.0 is\n"
        "'indistinguishable from ordinary content' and a step edge would sit well above.\n"
        "Fidelity is not measured here -- run tools/fidelity_probe.py for that; the\n"
        "agreement column is shown only because it is free, and it prefers blur."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
