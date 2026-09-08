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
I32 = npt.NDArray[np.int32]
F64 = npt.NDArray[np.float64]
U8 = npt.NDArray[np.uint8]
Panorama = npt.NDArray[np.uint8] | npt.NDArray[np.uint16]

__all__ = [
    "BIT_DEPTHS",
    "DEFAULT_FEATHER_POWER",
    "DEFAULT_SAMPLER",
    "SAMPLERS",
    "BandStats",
    "StitchResult",
    "bilinear_blend",
    "catmull_rom_weights",
    "cubic_blend",
    "stitch_bands",
    "stitch_frame",
]

SAMPLERS = ("nearest", "bilinear", "catmullrom")
DEFAULT_SAMPLER = "catmullrom"
"""How a tile is read at a fractional position.

`nearest` is P1's, kept because it is the byte-exact baseline the LUT is verified against
(AGENTS.md section 9, metric D) and because it is the cheapest preview.

The choice of the other two is measured, not conventional: `tools/resample_probe.py`
reports that at native density the map **magnifies** almost everywhere -- the minor scale
factor is below 1 in every latitude band (median 0.82 at the equator, 0.06 at the pole
cap) and only 0.1% of directions are minified in both axes. Under magnification there is
nothing to prefilter, so the mip/EWA machinery P4 was originally sketched with would only
blur; what nearest actually costs is up to half a pixel of positional error, which shows
up as replicated blocks. **Interpolation** is the fix, and the only question left is how
sharp a reconstruction to use.

`bilinear` is the cheap answer and `catmullrom` the faithful one -- a 4x4 cubic
(Catmull-Rom, a = -0.5), which is interpolating (it passes through the samples) and
noticeably sharper than bilinear under magnification, at four times the taps. It is the
default because the master is what P4 is for; `--sampler bilinear` is there when
throughput matters more.
"""

BIT_DEPTHS = (8, 16)
"""Output depth. 8 is delivery; 16 is a master, and only a master (P4 item 6).

The source is 8-bit, so 16 bits carry no extra information *from the render* -- what they
keep is the sub-level precision the resample and the blend produce, which 8-bit
quantisation throws away. That matters for a master a later stage will resample again,
and it is what makes an eventual 16-bit or EXR source a drop-in rather than a rewrite.

The scale factor is 257, not 256: it maps 255 exactly onto 65535, so every 8-bit level
lands on an exact multiple and an 8-bit and a 16-bit master of the same frame agree
wherever the blend happened to be integral.
"""

SIXTEEN_BIT_SCALE = np.float32(257.0)

LUMA_BT709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def quantise(blended: F32, bit_depth: int) -> Panorama:
    """Round a blended float32 panorama to the output depth.

    Round, do not truncate. `astype()` truncates toward zero, so a uniform input whose
    weighted average lands at 29.9999 would come out as 29 -- visible as banding on flat
    areas, and it doubles the average quantisation error.
    """
    if bit_depth == 8:
        return np.asarray(np.rint(np.clip(blended, 0.0, 255.0)), dtype=np.uint8)
    scaled = blended * SIXTEEN_BIT_SCALE
    return np.asarray(np.rint(np.clip(scaled, 0.0, 65535.0)), dtype=np.uint16)


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

    image: Panorama
    stats: BandStats

    @property
    def max_contributors(self) -> int:
        return int(self.stats.count.max()) if self.stats.count.size else 0

    @property
    def overlap_fraction(self) -> float:
        if not self.stats.count.size:
            return 0.0
        return float((self.stats.count >= 2).mean())


def catmull_rom_weights(f: F32) -> tuple[F32, F32, F32, F32]:
    """The four Catmull-Rom weights at fraction `f`, for taps at -1, 0, +1, +2.

    The classic cubic with a = -0.5: it interpolates, so a sample lands exactly on the
    source pixel when `f` is 0, and it has a mild negative lobe, which is what makes it
    sharper than bilinear -- and what lets it overshoot at a hard edge. The overshoot is
    clipped when the panorama is quantised, which is the accepted cost of the sharpness.

    Elementwise float32 in a fixed order, for metric D (see :func:`luma_bt709`).
    """
    half, one, two = np.float32(0.5), np.float32(1.0), np.float32(2.0)
    three, five = np.float32(1.5), np.float32(2.5)
    f2 = f * f
    f3 = f2 * f
    return (
        np.asarray(-half * f3 + f2 - half * f, dtype=np.float32),
        np.asarray(three * f3 - five * f2 + one, dtype=np.float32),
        np.asarray(-three * f3 + two * f2 + half * f, dtype=np.float32),
        np.asarray(half * f3 - half * f2, dtype=np.float32),
    )


def cubic_blend(rows: tuple[tuple[F32, F32, F32, F32], ...], fx: F32, fy: F32) -> F32:
    """Blend a 4x4 neighbourhood given as four rows of four taps, top row first.

    Separable: each row is collapsed horizontally, then the four results vertically. The
    order is fixed for the same reason :func:`bilinear_blend`'s is.
    """
    wx = catmull_rom_weights(fx)
    wy = catmull_rom_weights(fy)
    collapsed = [
        row[0] * wx[0][:, None]
        + row[1] * wx[1][:, None]
        + row[2] * wx[2][:, None]
        + row[3] * wx[3][:, None]
        for row in rows
    ]
    return np.asarray(
        collapsed[0] * wy[0][:, None]
        + collapsed[1] * wy[1][:, None]
        + collapsed[2] * wy[2][:, None]
        + collapsed[3] * wy[3][:, None],
        dtype=np.float32,
    )


def bilinear_blend(taps: tuple[F32, F32, F32, F32], fx: F32, fy: F32) -> F32:
    """Blend a 2x2 neighbourhood, `(top-left, top-right, bottom-left, bottom-right)`.

    Elementwise float32 in a fixed order, for the same reason :func:`luma_bt709` is: the
    LUT path and the per-frame path must produce identical bits, and anything that lets
    the summation order vary with array length breaks that.
    """
    top_left, top_right, bottom_left, bottom_right = taps
    one = np.float32(1.0)
    wx, wy = fx[:, None], fy[:, None]
    top = top_left * (one - wx) + top_right * wx
    bottom = bottom_left * (one - wx) + bottom_right * wx
    return np.asarray(top * (one - wy) + bottom * wy, dtype=np.float32)


DEFAULT_FEATHER_POWER = 2.0
"""Exponent on a tile's distance-to-edge when weighting the blend.

ROADMAP P4 item 4 asked for the feather *width* to be parameterised. The exponent is the
more useful knob, and the reason is geometric: distance-to-edge doubles as a proxy for
sampling quality, because a tile magnifies least near its own centre
(`tools/resample_probe.py`). A power of it therefore **prefers the tile that sees a
direction most centrally**, which a width -- flat interior, ramp near the border -- would
discard by making every interior weight equal.

The exponent also spans exactly the family P4 item 2 needs to ask about: as it rises the
blend slides from "average everything that can see this direction" toward "take the best
tile and ignore the rest", so sweeping it measures whether averaging misaligned tiles
costs detail. ROADMAP P4 records what the sweep found.
"""


def _feather(x: F64, y: F64, half: float, power: float = DEFAULT_FEATHER_POWER) -> F32:
    """Blend weight falling to zero at the tile border.

    Distance-to-edge in normalised tile coordinates raised to `power`: smooth enough that
    no seam is visible, cheap enough to recompute per frame, and it keeps the weight
    strictly positive inside so a direction covered by a single tile still resolves.

    The default is spelled as a squaring rather than a `pow`, so it stays bit-for-bit
    what P1 through P3 produced.
    """
    dx = 1.0 - np.abs(x) / half
    dy = 1.0 - np.abs(y) / half
    edge = np.minimum(dx, dy)
    raised = edge**2 if power == DEFAULT_FEATHER_POWER else edge**power
    return np.asarray(raised, dtype=np.float32) + _EPSILON


def stitch_bands(
    tiles: dict[int, U8],
    rig: Rig,
    width: int,
    height: int,
    *,
    band_rows: int = DEFAULT_BAND_ROWS,
    sampler: str = DEFAULT_SAMPLER,
    feather_power: float = DEFAULT_FEATHER_POWER,
    bit_depth: int = 8,
) -> StitchResult:
    """Blend `tiles` into a ``height x width x 3`` panorama, one horizontal band at a time.

    `tiles` maps a 1-based camera index to an ``(n, n, 3)`` uint8 array. Only the rig's
    unique indices are required; extra entries are ignored, so a caller may hand over all
    20 files or just the 15 distinct ones.
    """
    if width != 2 * height:
        raise ValueError(f"equirect output must be 2:1, got {width}x{height}")
    if sampler not in SAMPLERS:
        raise ValueError(f"sampler must be one of {SAMPLERS}, got {sampler!r}")
    if not 0.0 < feather_power <= 32.0:
        raise ValueError(f"feather_power must be in (0, 32], got {feather_power}")
    if bit_depth not in BIT_DEPTHS:
        raise ValueError(f"bit_depth must be one of {BIT_DEPTHS}, got {bit_depth}")
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
    image: Panorama = np.empty((height, width, 3), np.uint8 if bit_depth == 8 else np.uint16)
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
            tile = tiles[index]
            if sampler == "nearest":
                column, row = projection.tile_pixel_index(
                    x[visible], y[visible], tile_size, rig.fov_deg
                )
                sampled = tile[row, column].astype(np.float32)
            elif sampler == "bilinear":
                column, row, fx, fy = projection.tile_bilinear_taps(
                    x[visible], y[visible], tile_size, rig.fov_deg
                )
                sampled = bilinear_blend(
                    (
                        tile[row, column].astype(np.float32),
                        tile[row, column + 1].astype(np.float32),
                        tile[row + 1, column].astype(np.float32),
                        tile[row + 1, column + 1].astype(np.float32),
                    ),
                    fx,
                    fy,
                )
            else:
                column, row, fx, fy = projection.tile_cubic_taps(
                    x[visible], y[visible], tile_size, rig.fov_deg
                )

                def tap_row(
                    dy: int, tile: U8 = tile, row: I32 = row, column: I32 = column
                ) -> tuple[F32, F32, F32, F32]:
                    return (
                        tile[row + dy, column].astype(np.float32),
                        tile[row + dy, column + 1].astype(np.float32),
                        tile[row + dy, column + 2].astype(np.float32),
                        tile[row + dy, column + 3].astype(np.float32),
                    )  # fmt: skip

                sampled = cubic_blend(
                    (tap_row(0), tap_row(1), tap_row(2), tap_row(3)), fx, fy
                )  # fmt: skip
            w = _feather(x[visible], y[visible], half, feather_power)
            colour[visible] += sampled * w[:, None]
            weight[visible] += w
            luma = luma_bt709(sampled)
            band.count[visible] += 1.0
            band.luma_sum[visible] += luma
            band.luma_sq_sum[visible] += luma.astype(np.float64) ** 2

        blended = np.asarray(colour / np.maximum(weight, _EPSILON)[:, None], dtype=np.float32)
        image[start:stop] = quantise(blended, bit_depth).reshape(stop - start, width, 3)
        stats = band if stats.count.size == 0 else stats.extend(band)

    return StitchResult(image=image, stats=stats)


def stitch_frame(
    tiles: dict[int, U8],
    rig: Rig,
    width: int,
    *,
    band_rows: int = DEFAULT_BAND_ROWS,
    sampler: str = DEFAULT_SAMPLER,
    feather_power: float = DEFAULT_FEATHER_POWER,
    bit_depth: int = 8,
) -> StitchResult:
    """:func:`stitch_bands` with the 2:1 height implied by `width`."""
    if width % 2:
        raise ValueError(f"equirect width must be even, got {width}")
    return stitch_bands(
        tiles,
        rig,
        width,
        width // 2,
        band_rows=band_rows,
        sampler=sampler,
        feather_power=feather_power,
        bit_depth=bit_depth,
    )
