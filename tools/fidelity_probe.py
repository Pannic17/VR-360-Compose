"""Which sampler *reconstructs the source* best? Round-trips the master back to a tile.

The agreement metric (metric A) cannot answer this. It measures how closely overlapping
tiles agree, so a blurrier reconstruction scores better simply by being smoother, and a
sharper kernel that overshoots at edges scores worse even where it is more faithful.
Measured: bilinear reaches a median of 0.484 and Catmull-Rom 0.524, which says nothing
about which one kept more of the render.

So this asks the question directly. The master is resampled *back* into a tile's own
pixel grid, along that tile's real ray directions, and compared with the tile the
renderer produced. Information the resample threw away cannot come back, so whichever
master reconstructs the tile best is the one that kept the most.

The return trip is bilinear in equirect space for every candidate, so it costs all of
them the same. It is only measured over the central half of each tile: further out the
master is a blend in which this tile is not dominant, and near the border the feather
weight has gone to zero, so a disagreement there is not the sampler's doing.

    python tools/fidelity_probe.py
    python tools/fidelity_probe.py --width 3840 --cameras 1 6 11
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import numpy.typing as npt

from vr_compose import io, source
from vr_compose.projection import camera_to_world, half_extent, tile_rays
from vr_compose.rig import rig_for
from vr_compose.stitch import SAMPLERS, stitch_frame

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]
Panorama = npt.NDArray[np.uint8] | npt.NDArray[np.uint16]

DEFAULT_ROOT = pathlib.Path("E:/22")
CENTRAL = 0.5
"""Fraction of the tile's half-extent to score over. Outside it this tile is not the
dominant contributor to the master, so a difference is the blend's, not the sampler's."""


def sample_equirect(master: Panorama, lon: F64, lat: F64) -> F32:
    """Bilinear sample of an equirect image at the given directions.

    Longitude wraps; latitude clamps. The same for every candidate master, so it adds a
    constant blur to all of them rather than biasing the comparison.
    """
    height, width = master.shape[:2]
    u = ((lon / (2.0 * np.pi)) + 0.5) * width - 0.5
    v = (0.5 - lat / np.pi) * height - 0.5
    x0 = np.floor(u)
    y0 = np.clip(np.floor(v), 0, height - 2)
    fx = (u - x0).astype(np.float32)[:, None]
    fy = (v - y0).astype(np.float32)[:, None]
    c0 = x0.astype(np.int64) % width
    c1 = (x0.astype(np.int64) + 1) % width
    r0 = y0.astype(np.int64)
    top = master[r0, c0].astype(np.float32) * (1.0 - fx) + master[r0, c1].astype(np.float32) * fx
    bottom = (
        master[r0 + 1, c0].astype(np.float32) * (1.0 - fx)
        + master[r0 + 1, c1].astype(np.float32) * fx
    )
    return np.asarray(top * (1.0 - fy) + bottom * fy, dtype=np.float32)


def psnr(a: F32, b: F32) -> float:
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0.0 else 10.0 * np.log10(255.0**2 / mse)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=DEFAULT_ROOT)
    parser.add_argument("--frame", type=int, default=1656)
    parser.add_argument("--width", type=int, default=7680, help="master width; native is fairest")
    parser.add_argument(
        "--feathers",
        type=float,
        nargs="+",
        default=None,
        help="sweep feather exponents at the default sampler instead of sweeping samplers",
    )
    parser.add_argument(
        "--cameras",
        type=int,
        nargs="+",
        default=None,
        help="which tiles to score against (default: one per elevation ring)",
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

    # one tile per elevation ring by default: the regimes differ by latitude
    if args.cameras:
        cameras = [c for c in args.cameras if c in rig.unique_indices]
    else:
        by_elevation: dict[float, int] = {}
        for index in rig.unique_indices:
            by_elevation.setdefault(rig.view_for(index).elevation, index)
        cameras = sorted(by_elevation.values())

    half = half_extent(rig.fov_deg)
    rays = tile_rays(size, rig.fov_deg, mirrored=rig.mirrored)
    grid = (np.arange(size, dtype=np.float64) + 0.5) / size * 2.0 - 1.0
    gx, gy = np.meshgrid(grid, grid)
    central = ((np.abs(gx) < CENTRAL) & (np.abs(gy) < CENTRAL)).reshape(-1)

    print(
        f"frame {args.frame}, master {args.width}x{args.width // 2}, "
        f"tiles {cameras}, scored over the central {CENTRAL:.0%} of each tile\n"
    )
    label = "feather" if args.feathers else "sampler"
    print(f"  {label:<11} {'stitch':>8}  " + "  ".join(f"cam{c:<7}" for c in cameras) + "  mean")
    variants: list[tuple[str, dict[str, object]]] = (
        [(f"{power:g}", {"feather_power": power}) for power in args.feathers]
        if args.feathers
        else [(sampler, {"sampler": sampler}) for sampler in SAMPLERS]
    )
    for name, options in variants:
        started = time.time()
        master = stitch_frame(tiles, rig, args.width, **options).image  # type: ignore[arg-type]
        elapsed = time.time() - started
        scores = []
        for camera in cameras:
            view = rig.view_for(camera)
            world = camera_to_world(view.yaw, view.elevation) @ rays
            lon = np.arctan2(world[1], world[0])
            lat = np.arcsin(np.clip(world[2], -1.0, 1.0))
            reconstructed = sample_equirect(master, lon[central], lat[central])
            original = tiles[camera].reshape(-1, 3)[central].astype(np.float32)
            scores.append(psnr(reconstructed, original))
        print(
            f"  {name:<11} {elapsed:7.1f}s  "
            + "  ".join(f"{s:7.2f} dB" for s in scores)
            + f"  {float(np.mean(scores)):7.2f} dB"
        )
    print(
        "\nHigher is better: it is the master reconstructing the render it came from.\n"
        f"(half extent {half:g}; the return trip is bilinear for every row, so it is a\n"
        "constant cost, not a bias.)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
