"""Sphere coverage and sampling density of the solved rig. Reproduces AGENTS.md §3.

Pure geometry -- reads no image data. Answers two questions:

* Is the sphere fully covered, and with how much overlap? (drives the blend weights and
  the max-contributors constant K in the P1 LUT)
* How badly is each direction under-sampled relative to a tile centre? A pinhole tile's
  angular density falls as cos^3(off-axis angle), and the equirect output grid is sized
  to the tile *centre* density, so anything below 1.0 is upsampled.

    python tools/coverage_map.py
    python tools/coverage_map.py --width 2880 --out-dir scratch/
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import numpy.typing as npt
from PIL import Image

from vr_compose.projection import camera_to_world, equirect_directions
from vr_compose.rig import rig_for

EQUIRECT_W = 7680
TILE_PX = 1920
FOV_DEG = 90.0

RIG = rig_for(20)

F32 = npt.NDArray[np.float32]
I16 = npt.NDArray[np.int16]


def analyse(width: int, height: int, fov_deg: float) -> tuple[I16, F32]:
    """Per-direction (number of covering tiles, best relative sampling density)."""
    dirs = equirect_directions(width, height)
    t = np.tan(np.radians(fov_deg) / 2)
    cover = np.zeros(width * height, np.int16)
    density = np.zeros(width * height, np.float32)
    for yaw, elevation in ((v.yaw, v.elevation) for v in RIG.unique_views.values()):
        cam = camera_to_world(yaw, elevation).T @ dirs
        with np.errstate(divide="ignore", invalid="ignore"):
            x, y = cam[1] / cam[0], -cam[2] / cam[0]
        visible = (cam[0] > 0) & (np.abs(x) <= t) & (np.abs(y) <= t)
        cover += visible.astype(np.int16)
        cos_off = np.clip(cam[0] / np.linalg.norm(cam, axis=0), 0.0, 1.0)
        density = np.maximum(density, np.where(visible, cos_off**3, 0.0).astype(np.float32))
    return cover.reshape(height, width), density.reshape(height, width)


def solid_angle_weights(width: int, height: int) -> F32:
    lat = (0.5 - (np.arange(height, dtype=np.float64) + 0.5) / height) * np.pi
    return np.asarray(np.repeat(np.cos(lat)[:, None], width, axis=1), np.float32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1440, help="analysis width (2:1)")
    parser.add_argument("--fov", type=float, default=FOV_DEG)
    parser.add_argument(
        "--out-dir", type=pathlib.Path, default=None, help="write coverage.png / density.png here"
    )
    args = parser.parse_args(argv)

    width, height = args.width, args.width // 2
    cover, density = analyse(width, height, args.fov)
    weights = solid_angle_weights(width, height)
    total = float(weights.sum())

    print(
        f"analysis grid {width}x{height}, fov={args.fov} deg, "
        f"rig {RIG.name} with {len(RIG.unique_views)} distinct views\n"
    )
    print("solid-angle weighted coverage distribution:")
    for k in range(int(cover.max()) + 1):
        mask = cover == k
        if mask.any():
            share = float(weights[mask].sum()) / total
            label = "UNCOVERED" if k == 0 else f"{k} tile(s)"
            print(f"  {label:>11}: {share:8.4%}")
    print(
        f"\nmean tiles per direction: {float((cover * weights).sum()) / total:.2f}"
        f"   max: {int(cover.max())}  (K for the P1 LUT)"
    )

    seen = cover > 0
    print(
        f"\nbest-tile sampling density relative to a tile centre "
        f"(output grid is {EQUIRECT_W}/360 = {EQUIRECT_W / 360:.2f} px/deg "
        f"= tile centre {TILE_PX}/{args.fov:.0f} deg):"
    )
    print(
        f"  min={density[seen].min():.3f}  p5={np.percentile(density[seen], 5):.3f}  "
        f"median={np.percentile(density[seen], 50):.3f}  max={density[seen].max():.3f}"
    )
    print("\nby latitude:")
    print(f"  {'lat':>5} {'cover':>9} {'density':>15}")
    for lat_deg in (89, 80, 70, 60, 45, 30, 15, 0):
        row = min(height - 1, max(0, int((0.5 - lat_deg / 180) * height)))
        print(
            f"  {lat_deg:>5} {int(cover[row].min())}..{int(cover[row].max()):<6} "
            f"{density[row].min():.3f}..{density[row].max():.3f}"
        )
    pole = np.abs(np.arange(height) - height / 2) > height * 70 / 180
    print(
        f"\npole caps (|lat| > 70 deg): density min={density[pole].min():.3f} "
        f"mean={density[pole].mean():.3f} -> under-sampled "
        f"{1 / density[pole].mean():.1f}x vs a tile centre. This is the quality floor "
        f"and it is an upstream sampling problem, not a codec one (AGENTS.md §5)."
    )

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray((np.clip(cover, 0, 4) * 63).astype(np.uint8)).save(
            args.out_dir / "coverage.png"
        )
        Image.fromarray((density * 255).astype(np.uint8)).save(args.out_dir / "density.png")
        print(f"\nwrote {args.out_dir / 'coverage.png'} and {args.out_dir / 'density.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
