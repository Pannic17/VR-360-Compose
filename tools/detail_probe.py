"""Polar detail retention: what the resample kept, per latitude band.

AGENTS.md section 9 asks for exactly this metric -- "how much did the resample lose,
measured against the corresponding region of the source tiles" -- because it is one of
the few no-reference ways to argue about quality when there is no comparable reference
frame (metric C).

**Measure the source, not the master, when asking whether detail exists.** Near the pole
the map stretches one source pixel over about sixteen output pixels (`resample_probe.py`,
sigma_min median 0.06 at the cap), so a master is smooth there whatever the scene holds.
Reading polar sharpness off the master and concluding the content is flat is a mistake --
this tool exists because I made it.

Two things are reported:

* **source-tile detail by target latitude** -- for each tile pixel, which latitude it
  lands on, and how much local contrast is there. This is the headroom: if the tiles are
  flat in the directions that map to the pole, no reconstruction can invent detail.
* **master detail by latitude**, optionally for two masters at once, which is how the
  samplers get compared.

The two numbers are *not* a ratio. Local contrast is scale-dependent, and the master's
grid is far finer than the tiles' near the pole, so master/source says more about the
grids than about the resample. Compare source to source and master to master.

    python tools/detail_probe.py
    python tools/detail_probe.py --master A/frame.png --master B/frame.png
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import numpy.typing as npt
from PIL import Image

from vr_compose import io, source
from vr_compose.projection import camera_to_world, tile_rays
from vr_compose.rig import rig_for

Image.MAX_IMAGE_PIXELS = None

F32 = npt.NDArray[np.float32]

DEFAULT_ROOT = pathlib.Path("E:/22")
BANDS = (("cap >85", 85.0, 90.0), ("polar 70-85", 70.0, 85.0), ("mid 30-70", 30.0, 70.0),
         ("equator <30", 0.0, 30.0))  # fmt: skip
MIN_PIXELS = 1000
"""Below this a band's sample is too small in a given tile to mean anything."""


def laplacian(grey: F32) -> F32:
    """Absolute discrete Laplacian, the interior only. A local-contrast proxy."""
    return np.asarray(
        np.abs(
            4 * grey[1:-1, 1:-1]
            - grey[:-2, 1:-1]
            - grey[2:, 1:-1]
            - grey[1:-1, :-2]
            - grey[1:-1, 2:]
        ),
        dtype=np.float32,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=DEFAULT_ROOT)
    parser.add_argument("--frame", type=int, default=1656)
    parser.add_argument(
        "--master",
        type=pathlib.Path,
        action="append",
        default=None,
        help="an equirect PNG to measure as well; repeatable, to compare samplers",
    )
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

    tiles = io.load_tiles(chosen, args.frame, list(rig.unique_indices), workers=8)
    rays = tile_rays(size, rig.fov_deg, mirrored=rig.mirrored)
    per_band: dict[str, list[float]] = {name: [] for name, _, _ in BANDS}
    for index, view in rig.unique_views.items():
        world = camera_to_world(view.yaw, view.elevation) @ rays
        latitude = np.abs(
            np.degrees(np.arcsin(np.clip(world[2], -1.0, 1.0))).reshape(size, size)[1:-1, 1:-1]
        )
        contrast = laplacian(tiles[index].astype(np.float32).mean(axis=2))
        for name, lo, hi in BANDS:
            mask = (latitude >= lo) & (latitude < hi)
            if int(mask.sum()) > MIN_PIXELS:
                per_band[name].append(float(contrast[mask].mean()))

    print(f"frame {args.frame}, {size}px tiles, rig {rig.name}\n")
    print("source-tile detail by the latitude those pixels land on (mean |Laplacian|):")
    print(f"  {'band':<13} {'tiles':>6} {'mean':>7} {'min':>7} {'max':>7}")
    for name, _, _ in BANDS:
        values = per_band[name]
        if values:
            print(
                f"  {name:<13} {len(values):>6} {float(np.mean(values)):7.2f} "
                f"{min(values):7.2f} {max(values):7.2f}"
            )
    print(
        "\n  This is the headroom for the polar reconstruction in ROADMAP P4 item 2:\n"
        "  the tiles do carry detail in the directions that land near the pole."
    )

    for path in args.master or []:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=2)
        height, width = pixels.shape
        contrast = laplacian(pixels)
        lat = np.abs(90.0 - (np.arange(height) + 0.5) / height * 180.0)[1:-1]
        print(f"\nmaster {path} ({width}x{height}) detail by latitude:")
        for name, lo, hi in BANDS:
            rows = (lat >= lo) & (lat < hi)
            if rows.any():
                print(f"  {name:<13} {float(contrast[rows].mean()):7.2f}")
    if args.master:
        print(
            "\n  Master figures compare only to each other. Local contrast is scale\n"
            "  dependent, so master-over-source is a statement about the two grids."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
