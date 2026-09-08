"""Does blending and interpolating in linear light change the master? Measures it.

ROADMAP P4 item 3 asserts that "blending must happen in linear light, otherwise overlap
regions come out systematically dark". That is two claims, and they have very different
expected sizes:

* **The blend.** Overlapping tiles see the same radiance through the same nodal point,
  so they agree to about half a level (AGENTS.md section 9). Averaging two nearly equal
  numbers gives the same answer in any space -- the encoded-vs-linear gap grows with the
  *square* of the disagreement -- so this should be nearly a no-op, and if it is, the
  ROADMAP's stated reason for the change does not apply here.
* **The interpolation.** Bilinear across a high-contrast edge is where encoded-space
  arithmetic really does bias the result: the midpoint of a 0..255 edge is 128 encoded
  but about 188 if the average is taken in linear light. How much that matters depends
  entirely on how much high-contrast detail the content has.

So this measures the two separately, on real tiles, rather than assuming either.

    python tools/linear_light_probe.py
    python tools/linear_light_probe.py --width 7680 --frame 1660
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import numpy.typing as npt

from vr_compose import io, source
from vr_compose.projection import (
    equirect_directions,
    half_extent,
    project_to_tile,
    tile_bilinear_taps,
    tile_pixel_index,
)
from vr_compose.rig import Rig, rig_for
from vr_compose.stitch import _EPSILON, _feather, bilinear_blend

F32 = npt.NDArray[np.float32]
U8 = npt.NDArray[np.uint8]

DEFAULT_ROOT = pathlib.Path("E:/22")


def srgb_decode_table() -> F32:
    """8-bit code -> linear, as a 256-entry table. The decode is therefore free."""
    code = np.arange(256, dtype=np.float64) / 255.0
    linear = np.where(code <= 0.04045, code / 12.92, ((code + 0.055) / 1.055) ** 2.4)
    return np.asarray(linear, dtype=np.float32)


def srgb_encode(linear: F32) -> F32:
    """Linear -> 0..255 code, the sRGB OETF. Not a table: the input is continuous."""
    clipped = np.clip(linear, 0.0, 1.0)
    low = clipped * 12.92
    high = 1.055 * np.power(np.maximum(clipped, 1e-8), 1.0 / 2.4) - 0.055
    return np.asarray(np.where(clipped <= 0.0031308, low, high) * 255.0, dtype=np.float32)


def stitch(
    tiles: dict[int, U8], rig: Rig, width: int, *, sampler: str, linear: bool
) -> tuple[U8, F32]:
    """One panorama, plus the per-pixel count of contributing tiles.

    A deliberate re-implementation of the blend rather than a call into
    `vr_compose.stitch`: the point is to vary the *space* the arithmetic happens in,
    which the product code does not offer, and doing it here keeps the experiment from
    driving the design before there is a number to justify it.
    """
    height = width // 2
    half = half_extent(rig.fov_deg)
    table = srgb_decode_table()
    tile_size = next(iter(tiles.values())).shape[0]
    image = np.empty((height, width, 3), np.uint8)
    counts = np.zeros(width * height, np.float32)

    rows = 512
    for start in range(0, height, rows):
        stop = min(start + rows, height)
        dirs = equirect_directions(width, height, rows=slice(start, stop))
        pixels = dirs.shape[1]
        colour = np.zeros((pixels, 3), np.float32)
        weight = np.zeros(pixels, np.float32)

        for index, view in rig.unique_views.items():
            x, y, visible = project_to_tile(
                dirs, view.yaw, view.elevation, rig.fov_deg, mirrored=rig.mirrored
            )
            if not visible.any():
                continue
            tile = tiles[index]
            source_values = table[tile] if linear else tile.astype(np.float32)
            if sampler == "nearest":
                column, row = tile_pixel_index(x[visible], y[visible], tile_size, rig.fov_deg)
                sampled = source_values[row, column].astype(np.float32)
            else:
                column, row, fx, fy = tile_bilinear_taps(
                    x[visible], y[visible], tile_size, rig.fov_deg
                )
                sampled = bilinear_blend(
                    (
                        source_values[row, column].astype(np.float32),
                        source_values[row, column + 1].astype(np.float32),
                        source_values[row + 1, column].astype(np.float32),
                        source_values[row + 1, column + 1].astype(np.float32),
                    ),
                    fx,
                    fy,
                )
            w = _feather(x[visible], y[visible], half)
            colour[visible] += sampled * w[:, None]
            weight[visible] += w
            counts[start * width : stop * width][visible] += 1.0

        blended = np.asarray(colour / np.maximum(weight, _EPSILON)[:, None], dtype=np.float32)
        codes = srgb_encode(blended) if linear else blended
        image[start:stop] = (
            np.rint(np.clip(codes, 0.0, 255.0)).astype(np.uint8).reshape(stop - start, width, 3)
        )
    return image, counts


def compare(name: str, a: U8, b: U8, counts: F32, height: int, width: int) -> None:
    difference = np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.float32)
    signed = (b.astype(np.float32) - a.astype(np.float32)).mean()
    overlap = (counts >= 2).reshape(height, width)
    single = ~overlap
    print(f"{name}")
    p99, p999 = np.percentile(difference, 99), np.percentile(difference, 99.9)
    print(
        f"  all pixels     mean {difference.mean():7.4f}  p99 {p99:6.2f}  "
        f"p99.9 {p999:6.2f}  max {difference.max():6.0f}  signed {signed:+.4f}"
    )
    for label, mask in (("overlap >=2", overlap), ("single tile", single)):
        if not mask.any():
            continue
        part = difference[mask]
        print(
            f"  {label:<14} mean {part.mean():7.4f}  p99 {np.percentile(part, 99):6.2f}  "
            f"max {part.max():6.0f}   ({float(mask.mean()):.1%} of pixels)"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=DEFAULT_ROOT)
    parser.add_argument("--frame", type=int, default=1656)
    parser.add_argument("--width", type=int, default=3840, help="analysis width (2:1)")
    args = parser.parse_args(argv)

    found = [s for s in source.scan(args.root) if s.usable]
    if not found:
        print(f"no usable source set in {args.root}", file=sys.stderr)
        return 2
    chosen = found[0]
    rig = rig_for(chosen.camera_count)
    tiles = io.load_tiles(chosen, args.frame, list(rig.unique_indices), workers=8)
    height = args.width // 2
    print(f"frame {args.frame}, {args.width}x{height}, sRGB transfer assumed for the source\n")

    versions = {
        (sampler, linear): stitch(tiles, rig, args.width, sampler=sampler, linear=linear)
        for sampler in ("nearest", "bilinear")
        for linear in (False, True)
    }
    counts = versions[("bilinear", False)][1]

    print("=== the blend alone (nearest sampling, so no interpolation is involved) ===")
    compare(
        "encoded blend -> linear blend",
        versions[("nearest", False)][0],
        versions[("nearest", True)][0],
        counts,
        height,
        args.width,
    )
    print("\n=== blend + interpolation (bilinear) ===")
    compare(
        "encoded -> linear",
        versions[("bilinear", False)][0],
        versions[("bilinear", True)][0],
        counts,
        height,
        args.width,
    )
    print("\n=== for scale: what the sampler change itself did (both encoded) ===")
    compare(
        "nearest -> bilinear",
        versions[("nearest", False)][0],
        versions[("bilinear", False)][0],
        counts,
        height,
        args.width,
    )
    print(
        "\nRead the 'overlap' vs 'single tile' split: a difference that shows up in "
        "single-tile\nregions too is the interpolation, not the blend -- only overlap "
        "regions are blended."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
