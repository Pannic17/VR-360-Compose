"""What the equirect <- tile resample actually does to a pixel. Decides P4's filter.

`coverage_map.py` answers "how many tiles see this direction, and how finely do they
sample it". That is not enough to choose a resampling filter, because a filter has to
know, per output pixel, whether the map is *minifying* the source (several source pixels
fall inside one output pixel -- aliasing, needs a prefilter) or *magnifying* it (one
source pixel spans several output pixels -- needs a good interpolation kernel), and how
anisotropic that is.

So this measures the Jacobian of the map directly, by finite differences on the real
projection code: for an output pixel, where do its two neighbours land in tile pixel
coordinates? The singular values of that 2x2 give the two principal scale factors.

    sigma > 1   one output step crosses more than one source pixel -> MINIFYING
    sigma < 1   one source pixel covers more than one output pixel -> MAGNIFYING
    sigma_max / sigma_min   how elliptical the footprint is

Nearest neighbour is wrong in both regimes, but differently: it aliases under
minification and it replicates blocks under magnification. Which regime dominates, and
where, is what this script is for.

    python tools/resample_probe.py
    python tools/resample_probe.py --width 2880 --tile 3840   # a 16K render
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import numpy.typing as npt

from vr_compose.projection import camera_to_world, half_extent
from vr_compose.rig import Rig, rig_for

F64 = npt.NDArray[np.float64]
Bool = npt.NDArray[np.bool_]

DEFAULT_TILE = 1920
DEFAULT_FOV = 90.0


def _directions(width: int, height: int, du: float = 0.0, dv: float = 0.0) -> F64:
    """Unit directions for equirect pixel centres offset by `(du, dv)` pixels."""
    lon = ((np.arange(width, dtype=np.float64) + 0.5 + du) / width - 0.5) * 2.0 * np.pi
    lat = (0.5 - (np.arange(height, dtype=np.float64) + 0.5 + dv) / height) * np.pi
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    cos_lat = np.cos(lat_grid)
    dirs = np.stack([cos_lat * np.cos(lon_grid), cos_lat * np.sin(lon_grid), np.sin(lat_grid)])
    return np.asarray(dirs.reshape(3, -1), dtype=np.float64)


def _to_tile_pixels(
    dirs: F64, yaw: float, elevation: float, fov_deg: float, tile: int, *, mirrored: bool
) -> tuple[F64, F64, Bool]:
    """World directions -> continuous tile pixel coordinates, plus visibility."""
    t = half_extent(fov_deg)
    cam = camera_to_world(yaw, elevation).T @ dirs
    forward = cam[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = cam[1] / forward
        y = -cam[2] / forward
    if not mirrored:
        x = -x
    visible = (forward > 0.0) & (np.abs(x) <= t) & (np.abs(y) <= t)
    # the same mapping `tile_pixel_index` rounds, kept continuous
    return (x / t + 1.0) * 0.5 * tile, (y / t + 1.0) * 0.5 * tile, visible


def scale_factors(
    rig: Rig, width: int, height: int, tile: int, fov_deg: float
) -> tuple[F64, F64, F64]:
    """Per output pixel: `(sigma_max, sigma_min, tiles seen)` for the *finest* tile.

    "Finest" means the tile whose footprint is smallest, i.e. the one a sampler would
    most want to read -- picking the maximum sigma would describe the worst contributor
    rather than the achievable best.
    """
    base = _directions(width, height)
    right = _directions(width, height, du=1.0)
    down = _directions(width, height, dv=1.0)
    pixels = width * height
    best = np.full(pixels, np.inf)
    other = np.full(pixels, np.inf)
    seen = np.zeros(pixels)

    for view in rig.unique_views.values():
        args = (view.yaw, view.elevation, fov_deg, tile)
        x0, y0, visible = _to_tile_pixels(base, *args, mirrored=rig.mirrored)
        if not visible.any():
            continue
        x1, y1, _ = _to_tile_pixels(right, *args, mirrored=rig.mirrored)
        x2, y2, _ = _to_tile_pixels(down, *args, mirrored=rig.mirrored)
        # columns of the Jacobian: source pixels moved per one output pixel step
        a, b = x1 - x0, x2 - x0
        c, d = y1 - y0, y2 - y0
        # singular values of [[a, b], [c, d]] without assembling matrices
        e = (a * a + b * b + c * c + d * d) / 2.0
        f = np.sqrt(
            np.maximum(((a * a + b * b - c * c - d * d) / 2.0) ** 2 + (a * c + b * d) ** 2, 0.0)
        )
        hi = np.sqrt(np.maximum(e + f, 0.0))
        lo = np.sqrt(np.maximum(e - f, 0.0))
        seen += visible
        # keep the tile with the smallest major axis
        better = visible & (hi < best)
        best = np.where(better, hi, best)
        other = np.where(better, lo, other)
    return best, other, seen


def _bands(height: int) -> list[tuple[str, Bool]]:
    lat = 90.0 - (np.arange(height, dtype=np.float64) + 0.5) / height * 180.0
    absolute = np.abs(lat)
    return [
        ("equator |lat|<30", absolute < 30),
        ("mid 30-70", (absolute >= 30) & (absolute < 70)),
        ("polar 70-85", (absolute >= 70) & (absolute < 85)),
        ("cap |lat|>85", absolute >= 85),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1440, help="analysis width (2:1)")
    parser.add_argument(
        "--output-width",
        type=int,
        default=None,
        help="the real output width the scale factors describe (default: --width)",
    )
    parser.add_argument("--tile", type=int, default=DEFAULT_TILE)
    parser.add_argument("--fov", type=float, default=DEFAULT_FOV)
    parser.add_argument("--cameras", type=int, default=20)
    parser.add_argument("--out-dir", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)

    rig = rig_for(args.cameras)
    analysis_width = args.width
    height = analysis_width // 2
    output_width = args.output_width or rig.native_width(args.tile)
    # Measure on a coarse grid, then rescale: the Jacobian is linear in the output
    # step, so sigma at width W is sigma(analysis) * analysis_width / W.
    rescale = analysis_width / output_width

    hi, lo, seen = scale_factors(rig, analysis_width, height, args.tile, args.fov)
    hi, lo = hi * rescale, lo * rescale
    covered = np.isfinite(hi) & (seen > 0)

    print(
        f"rig {rig.name}, tile {args.tile}px @ {args.fov:g}deg, output {output_width}x"
        f"{output_width // 2} (native width for this tile: {rig.native_width(args.tile)})\n"
    )
    print("source pixels crossed per output pixel step, for the finest tile seeing it:")
    print("  sigma > 1 = MINIFYING (aliasing risk)   sigma < 1 = MAGNIFYING (interpolation)\n")
    print(
        f"  {'band':<18} {'sigma_max':>19} {'sigma_min':>19} {'anisotropy':>11} {'minifying':>10}"
    )
    for name, rows in _bands(height):
        mask = np.zeros((height, analysis_width), bool)
        mask[rows] = True
        band = mask.reshape(-1) & covered
        if not band.any():
            continue
        big, small = hi[band], lo[band]
        print(
            f"  {name:<18} "
            f"{big.min():5.2f} {np.median(big):5.2f} {big.max():6.2f} "
            f"{small.min():5.2f} {np.median(small):5.2f} {small.max():6.2f} "
            f"{np.median(big / np.maximum(small, 1e-9)):10.2f} "
            f"{float((big > 1.0).mean()):9.1%}"
        )
    print("  (each triple is min / median / max)\n")

    share = float((hi[covered] > 1.0).mean())
    print(
        f"overall: {share:.1%} of covered directions are minified in their major axis, "
        f"{float((lo[covered] > 1.0).mean()):.1%} in both axes.\n"
        f"median anisotropy {np.median(hi[covered] / np.maximum(lo[covered], 1e-9)):.2f}, "
        f"worst {float((hi[covered] / np.maximum(lo[covered], 1e-9)).max()):.1f}"
    )

    if args.out_dir:
        from PIL import Image

        args.out_dir.mkdir(parents=True, exist_ok=True)
        for name, field in (("sigma_max", hi), ("sigma_min", lo)):
            picture = np.clip(np.nan_to_num(field, nan=0.0, posinf=0.0), 0.0, 2.0) / 2.0
            Image.fromarray((picture.reshape(height, analysis_width) * 255).astype(np.uint8)).save(
                args.out_dir / f"{name}.png"
            )
        print(f"\nwrote {args.out_dir / 'sigma_max.png'} and sigma_min.png (0..2 mapped to 0..255)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
