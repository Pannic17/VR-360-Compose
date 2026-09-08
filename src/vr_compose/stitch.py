"""Reproject the tiles of one frame into an equirectangular panorama.

Because the rig is a fixed, single-nodal-point arrangement (AGENTS.md §3), stitching is a
static resample plus a blend -- no calibration, no feature matching, no optical flow. The
overlapping tiles agree to a median of 0.70/255, so a feathered weighted average is
enough and there is no seam to carve.

Two structural choices worth keeping:

* **Band-wise output.** The full 7680x3840 direction array alone is 708 MB in float64, so
  the panorama is produced in horizontal bands. Peak memory then depends on the band
  height rather than the output size, which is also what P2's streaming path needs.
* **Statistics alongside the pixels.** :class:`BandStats` tracks per-direction tile
  agreement while it blends, so the geometry check in :mod:`vr_compose.verify` costs
  nothing extra.

P1 samples nearest-neighbour and recomputes geometry per frame. P2 replaces the geometry
with a cached LUT and P4 replaces the sampler; both must keep this module's output
identical on the same inputs, which is what the verification metrics are for.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import numpy.typing as npt

from vr_compose import projection
from vr_compose.rig import Rig

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]
U8 = npt.NDArray[np.uint8]

__all__ = ["BandStats", "StitchResult", "stitch_bands", "stitch_frame"]

LUMA_BT709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def luma_bt709(rgb: F32) -> F32:
    """Rec. 709 luma of an ``(n, 3)`` float32 array, computed elementwise on purpose.

    ``rgb @ LUMA_BT709`` would hand the dot product to BLAS, whose summation order depends
    on array length and alignment, so identical pixels could get luma differing in the
    last bit depending on how the work was batched. Elementwise float32 arithmetic is
    deterministic, which the plan-equals-stitch guarantee in :mod:`vr_compose.warp`
    depends on.
    """
    return np.asarray(
        rgb[:, 0] * LUMA_BT709[0] + rgb[:, 1] * LUMA_BT709[1] + rgb[:, 2] * LUMA_BT709[2],
        dtype=np.float32,
    )


DEFAULT_BAND_ROWS = 512
_EPSILON = np.float32(1e-6)


@dataclasses.dataclass(slots=True)
class BandStats:
    """Per-direction tile agreement, accumulated while blending.

    Holds sums rather than the finished statistic so bands can be combined.
    """

    count: F32
    luma_sum: npt.NDArray[np.float64]
    luma_sq_sum: npt.NDArray[np.float64]

    @classmethod
    def zeros(cls, pixels: int) -> BandStats:
        return cls(
            count=np.zeros(pixels, np.float32),
            luma_sum=np.zeros(pixels, np.float64),
            luma_sq_sum=np.zeros(pixels, np.float64),
        )

    def disagreement(self) -> npt.NDArray[np.float64]:
        """Standard deviation of tile luma, for directions seen by two or more tiles."""
        overlap = self.count >= 2
        n = self.count[overlap].astype(np.float64)
        mean = self.luma_sum[overlap] / n
        variance = np.maximum(self.luma_sq_sum[overlap] / n - mean * mean, 0.0)
        return np.asarray(np.sqrt(variance), dtype=np.float64)

    def extend(self, other: BandStats) -> BandStats:
        return BandStats(
            count=np.concatenate([self.count, other.count]),
            luma_sum=np.concatenate([self.luma_sum, other.luma_sum]),
            luma_sq_sum=np.concatenate([self.luma_sq_sum, other.luma_sq_sum]),
        )


@dataclasses.dataclass(slots=True)
class StitchResult:
    """A finished panorama and the statistics gathered while producing it."""

    image: U8
    stats: BandStats

    @property
    def max_contributors(self) -> int:
        return int(self.stats.count.max()) if self.stats.count.size else 0

    @property
    def overlap_fraction(self) -> float:
        if not self.stats.count.size:
            return 0.0
        return float((self.stats.count >= 2).mean())


def _feather(x: F64, y: F64, half: float) -> F32:
    """Blend weight falling to zero at the tile border.

    Squared distance-to-edge in normalised tile coordinates: smooth enough that no seam is
    visible, cheap enough to recompute per frame, and it keeps the weight strictly
    positive inside so a direction covered by a single tile still resolves.
    """
    dx = 1.0 - np.abs(x) / half
    dy = 1.0 - np.abs(y) / half
    return np.asarray(np.minimum(dx, dy) ** 2, dtype=np.float32) + _EPSILON


def stitch_bands(
    tiles: dict[int, U8],
    rig: Rig,
    width: int,
    height: int,
    *,
    band_rows: int = DEFAULT_BAND_ROWS,
) -> StitchResult:
    """Blend `tiles` into a ``height x width x 3`` panorama, one horizontal band at a time.

    `tiles` maps a 1-based camera index to an ``(n, n, 3)`` uint8 array. Only the rig's
    unique indices are required; extra entries are ignored, so a caller may hand over all
    20 files or just the 15 distinct ones.
    """
    if width != 2 * height:
        raise ValueError(f"equirect output must be 2:1, got {width}x{height}")
    views = rig.unique_views
    missing = sorted(set(views) - set(tiles))
    if missing:
        raise ValueError(f"missing tiles for camera indices {missing}")

    sizes = {tiles[index].shape[0] for index in views}
    if len(sizes) != 1:
        raise ValueError(f"tiles have mixed sizes: {sorted(sizes)}")
    tile_size = sizes.pop()
    for index in views:
        tile = tiles[index]
        if tile.ndim != 3 or tile.shape[0] != tile.shape[1] or tile.shape[2] != 3:
            raise ValueError(f"camera {index}: expected a square RGB tile, got {tile.shape}")

    half = projection.half_extent(rig.fov_deg)
    image = np.empty((height, width, 3), np.uint8)
    stats = BandStats.zeros(0)

    for start in range(0, height, max(1, band_rows)):
        stop = min(start + band_rows, height)
        dirs = projection.equirect_directions(width, height, rows=slice(start, stop))
        pixels = dirs.shape[1]
        colour = np.zeros((pixels, 3), np.float32)
        weight = np.zeros(pixels, np.float32)
        band = BandStats.zeros(pixels)

        for index, view in views.items():
            x, y, visible = projection.project_to_tile(
                dirs, view.yaw, view.elevation, rig.fov_deg, mirrored=rig.mirrored
            )
            if not visible.any():
                continue
            column, row = projection.tile_pixel_index(
                x[visible], y[visible], tile_size, rig.fov_deg
            )
            sampled = tiles[index][row, column].astype(np.float32)
            w = _feather(x[visible], y[visible], half)
            colour[visible] += sampled * w[:, None]
            weight[visible] += w
            luma = luma_bt709(sampled)
            band.count[visible] += 1.0
            band.luma_sum[visible] += luma
            band.luma_sq_sum[visible] += luma.astype(np.float64) ** 2

        blended = colour / np.maximum(weight, _EPSILON)[:, None]
        # Round, do not truncate. astype() truncates toward zero, so a uniform input
        # whose weighted average lands at 29.9999 would come out as 29 -- visible as
        # banding on flat areas, and it doubles the average quantisation error.
        quantised = np.rint(np.clip(blended, 0.0, 255.0)).astype(np.uint8)
        image[start:stop] = quantised.reshape(stop - start, width, 3)
        stats = band if stats.count.size == 0 else stats.extend(band)

    return StitchResult(image=image, stats=stats)


def stitch_frame(
    tiles: dict[int, U8], rig: Rig, width: int, *, band_rows: int = DEFAULT_BAND_ROWS
) -> StitchResult:
    """:func:`stitch_bands` with the 2:1 height implied by `width`."""
    if width % 2:
        raise ValueError(f"equirect width must be even, got {width}")
    return stitch_bands(tiles, rig, width, width // 2, band_rows=band_rows)
