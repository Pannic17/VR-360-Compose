"""Photometric harmonisation: take the smooth brightness differences between tiles out.

Cameras rendered from the same point do not always see the same thing. UE's volumetric
fog is integrated per view, motion blur is a screen-space effect, bloom and exposure are
per camera -- so two tiles that overlap can disagree by several levels over large, smooth
regions, and the blend then shows tile-shaped patches. The reference render (`E:/22`) had
none of this; `L_Cathedral_v004` had 10 % of its overlap disagreeing by more than 8 levels,
almost all of it fog and blur (ROADMAP P9, AGENTS.md section 9). The upstream settings
cannot be changed, so this module absorbs the difference here.

**What it does.** For every tile, the low-frequency part of (what this tile saw) minus
(what all tiles agree on) is estimated on a coarse angular grid and subtracted from the
tile's samples before they are weighted into the panorama. Edges and texture are left
alone: the correction is a smooth field, clamped to a few levels, and it is zero wherever
the tiles already agree (on the reference data no output pixel moves by more than 2).

**How.** The estimate runs on its own small warp plan (`COARSE_WIDTH` wide, bilinear), so
it costs a fraction of the real warp and is independent of the output size: a 4K and an 8K
render get the same corrections. Per iteration: blend the coarse tiles, bin each tile's
residual against that blend into `GRID` cells weighted like the blend, drop cells the
tile barely covers, fill the rest from their neighbours, blur, accumulate, clamp.

**Where it is applied.** :meth:`vr_compose.warp.WarpPlan.apply` and its GPU twin take the
grids and subtract :func:`vr_compose.warp.upsample_grid` of them from every sample -- the
same float32 arithmetic on both, so CPU and GPU stay byte-identical. With no corrections
the warp is exactly what P1 verified.

Defaults: **on** (user, 2026-09-10: the defaults must render the data sources we actually
have). `--no-harmonise` turns it off.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import numpy.typing as npt

from vr_compose import verify
from vr_compose.rig import Rig
from vr_compose.stitch import (
    _EPSILON,
    DEFAULT_FEATHER_POWER,
    BandStats,
    bilinear_blend,
    luma_bt709,
)
from vr_compose.warp import WarpPlan

__all__ = [
    "BLUR_RADIUS",
    "CLAMP",
    "COARSE_WIDTH",
    "DEFAULT_HARMONISE",
    "GRID",
    "ITERATIONS",
    "Corrections",
    "Harmoniser",
]

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]
U8 = npt.NDArray[np.uint8]

DEFAULT_HARMONISE = True

GRID = (60, 120)
"""Correction grid, `(rows, columns)`: 3 degrees a cell over the whole sphere. Angular, so
it does not depend on the output size, and coarse enough that only fog-scale differences
survive it."""

COARSE_WIDTH = 240
"""Width of the estimation plan: 2 output pixels a cell, 1/1024 of the 8K warp. Measured on
v004 frame 100 and the reference frame, 960 / 480 / 360 / 240 give the same corrections to
within 0.01 of every metric -- the 3-degree grid and the blur set the detail, not the sample
count -- while the time falls 1.05 -> 0.48 -> 0.38 -> 0.30 s a frame on the CPU. What is
left is fixed cost (blurs, fills, bincounts). It runs in the prefetch thread, beside the
decode, so on the CPU path it is hidden behind the 11 s warp entirely."""

CLAMP = 32.0
"""Largest correction, in 8-bit levels. The fog wedge in v004 frame 100 wants 24-32 (at 24
it was clipped and 2 % of the overlap stayed over 8 levels apart; at 32, 1 %). Anything
much larger would be hiding a real problem rather than smoothing a rendering one."""

BLUR_RADIUS = 3
"""Box-blur radius in cells, applied three times (about a Gaussian of 9 degrees). Measured
on v004 frame 100: radius 3 leaves 1.04 % of the overlap over 8 levels and moves 3.6 % of
output pixels by more than 2; radius 4 gives 1.09 % / 6.3 %, radius 6 1.30 % / 14 %.
Smaller follows the fog more closely and touches less of the picture."""

ITERATIONS = 2
"""Residuals are measured against a blend that includes the tile itself, so one pass under-
corrects; two converge (frame 100: 1.41 % -> 1.19 % over 8, a third pass changes nothing)."""

MIN_COVERAGE = 0.05
"""A cell a tile covers with less than this share of its best cell's weight is not
trusted: its mean is a few edge samples, not the tile's view of that direction."""


@dataclasses.dataclass(frozen=True, slots=True)
class Corrections:
    """One frame's correction grids, camera -> `(gh, gw, 3)` float32, and how they did."""

    grids: dict[int, F32]
    raw: verify.Agreement
    """Coarse-plan agreement before correcting -- the tiles as rendered."""
    corrected: verify.Agreement
    """Coarse-plan agreement after correcting; the full-resolution gate sees about this."""
    seconds: float

    @property
    def max_abs(self) -> float:
        return max((float(np.abs(g).max()) for g in self.grids.values()), default=0.0)

    def describe(self) -> str:
        return (
            f"harmonise: tiles disagreed median {self.raw.median:.2f} / >8 "
            f"{self.raw.fraction_over_8:.2%}, corrected to {self.corrected.median:.2f} / "
            f">8 {self.corrected.fraction_over_8:.2%}; largest correction "
            f"{self.max_abs:.1f} levels ({self.seconds * 1e3:.0f} ms)"
        )


class Harmoniser:
    """Estimates per-tile corrections for a rig, one frame at a time.

    Everything that depends only on the rig is precomputed here: the coarse plan, which
    grid cell every coarse contributor falls in, and the four grid corners and weights its
    correction is interpolated from. A frame then costs a handful of `bincount`s over the
    ~1.2 M coarse entries and some blurs on a `(tiles, 60, 120, 3)` stack -- a fraction of
    a second on the CPU for 1920-pixel tiles, against 11 s for the 8K warp it feeds.
    """

    def __init__(
        self,
        rig: Rig,
        tile_size: int,
        *,
        feather_power: float = DEFAULT_FEATHER_POWER,
        width: int = COARSE_WIDTH,
        grid: tuple[int, int] = GRID,
    ) -> None:
        if width % 2:
            raise ValueError(f"coarse width must be even, got {width}")
        self.plan = WarpPlan.build(
            rig, width, width // 2, tile_size, sampler="bilinear", feather_power=feather_power
        )
        self.grid = grid
        self.width, self.height = width, width // 2
        gh, gw = grid
        cells_per_tile = gh * gw
        self.cameras = [tile.camera for tile in self.plan.tiles]
        # concatenated over tiles, in plan order
        self._out_index = np.concatenate([t.out_index for t in self.plan.tiles])
        self._weight = np.concatenate([t.weight for t in self.plan.tiles]).astype(np.float64)
        cells, corners, corner_weights = [], [], []
        for k, tile in enumerate(self.plan.tiles):
            row = tile.out_index // width
            col = tile.out_index % width
            cy = np.minimum(row * gh // self.height, gh - 1)
            cx = np.minimum(col * gw // width, gw - 1)
            cells.append(k * cells_per_tile + cy * gw + cx)
            # bilinear corners of the correction at each contributor, offset into tile k's grid
            fy = (row.astype(np.float32) + np.float32(0.5)) * np.float32(gh / self.height) - 0.5
            fx = (col.astype(np.float32) + np.float32(0.5)) * np.float32(gw / width) - 0.5
            y0 = np.clip(np.floor(fy), 0, gh - 2).astype(np.int64)
            x0 = np.clip(np.floor(fx), 0, gw - 2).astype(np.int64)
            ty = np.clip(fy - y0, 0.0, 1.0).astype(np.float32)
            tx = np.clip(fx - x0, 0.0, 1.0).astype(np.float32)
            base = k * cells_per_tile + y0 * gw + x0
            corners.append(np.stack([base, base + 1, base + gw, base + gw + 1], axis=1))
            corner_weights.append(
                np.stack([(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty], axis=1)
            )
        self._cells = np.concatenate(cells).astype(np.int64)
        self._corners = np.concatenate(corners).astype(np.int64)
        self._corner_weights = np.concatenate(corner_weights).astype(np.float32)
        self._bins = len(self.plan.tiles) * cells_per_tile
        self._pixels = self.width * self.height

    @property
    def build_seconds(self) -> float:
        return self.plan.build_seconds

    def estimate(self, tiles: dict[int, U8]) -> Corrections:
        """The corrections for one frame's tiles."""
        started = time.perf_counter()
        plan = self.plan
        gh, gw = self.grid
        k_tiles = len(plan.tiles)
        step = plan.tile_size
        parts = []
        for tile in plan.tiles:
            source = tiles[tile.camera].reshape(-1, 3)
            src = tile.src_index
            parts.append(
                bilinear_blend(
                    (
                        source[src].astype(np.float32),
                        source[src + 1].astype(np.float32),
                        source[src + step].astype(np.float32),
                        source[src + step + 1].astype(np.float32),
                    ),
                    tile.fx,
                    tile.fy,
                )
            )
        samples = np.concatenate(parts)  # (N, 3) float32, tiles in plan order

        stack = np.zeros((k_tiles, gh, gw, 3), np.float64)  # the corrections, all tiles
        blended, raw = self._blend(samples, None, with_stats=True)
        corrected = raw
        for iteration in range(ITERATIONS):
            current = samples if iteration == 0 else samples - self._upsample(stack)
            residual = (current - blended[self._out_index]).astype(np.float64)
            sums = np.stack(
                [
                    np.bincount(
                        self._cells, weights=residual[:, c] * self._weight, minlength=self._bins
                    )
                    for c in range(3)
                ],
                axis=1,
            ).reshape(k_tiles, gh, gw, 3)
            counts = np.bincount(self._cells, weights=self._weight, minlength=self._bins)
            weights = counts.reshape(k_tiles, gh, gw)
            floor = MIN_COVERAGE * weights.reshape(k_tiles, -1).max(axis=1)
            mask = weights > floor[:, None, None]
            mean = sums / np.maximum(weights, 1e-9)[..., None]
            field = np.where(mask[..., None], mean, 0.0)
            field = _box_blur(_fill_smooth(field, mask), BLUR_RADIUS)
            stack = np.clip(stack + field, -CLAMP, CLAMP)
            last = iteration == ITERATIONS - 1
            blended, corrected = self._blend(samples, stack, with_stats=last)
        grids = {cam: stack[k].astype(np.float32) for k, cam in enumerate(self.cameras)}
        return Corrections(grids, raw, corrected, time.perf_counter() - started)

    def _upsample(self, stack: F64) -> F32:
        """Every contributor's correction: four gathers from the stacked grids."""
        flat = stack.reshape(-1, 3).astype(np.float32)
        corners, weights = self._corners, self._corner_weights
        total = flat[corners[:, 0]] * weights[:, 0:1]
        for k in (1, 2, 3):
            total += flat[corners[:, k]] * weights[:, k : k + 1]
        return np.asarray(total, dtype=np.float32)

    def _blend(
        self, samples: F32, stack: F64 | None, *, with_stats: bool
    ) -> tuple[F32, verify.Agreement]:
        """The coarse blend as `(pixels, 3)` float32, and (when asked) its agreement."""
        corrected = samples if stack is None else samples - self._upsample(stack)
        out = self._out_index
        w = self._weight
        colour = np.stack(
            [
                np.bincount(out, weights=corrected[:, c] * w, minlength=self._pixels)
                for c in range(3)
            ],
            axis=1,
        )
        weight = np.bincount(out, weights=w, minlength=self._pixels)
        blended = np.asarray(colour / np.maximum(weight, float(_EPSILON))[:, None], np.float32)
        if not with_stats:
            return blended, _NO_AGREEMENT
        luma = luma_bt709(corrected).astype(np.float64)
        stats = BandStats(
            count=np.bincount(out, minlength=self._pixels).astype(np.float32),
            luma_sum=np.bincount(out, weights=luma, minlength=self._pixels),
            luma_sq_sum=np.bincount(out, weights=luma * luma, minlength=self._pixels),
        )
        return blended, verify.agreement(stats)


_NO_AGREEMENT = verify.Agreement(0.0, 0.0, 0.0, 0.0, 0, 0)


def _box_blur(field: F64, radius: int) -> F64:
    """Three passes of a box blur over the two grid axes of a `(k, h, w, c)` stack."""
    out = field
    size = 2 * radius + 1
    for _ in range(3):
        padded = np.pad(out, ((0, 0), (radius, radius), (radius, radius), (0, 0)), mode="edge")
        summed = np.cumsum(np.cumsum(padded, axis=1), axis=2)
        summed = np.pad(summed, ((0, 0), (1, 0), (1, 0), (0, 0)))
        a = summed[:, size:, size:]
        b = summed[:, :-size, size:]
        c = summed[:, size:, :-size]
        d = summed[:, :-size, :-size]
        out = (a - b - c + d) / (size * size)
    return np.asarray(out, dtype=np.float64)


def _fill_smooth(field: F64, mask: npt.NDArray[np.bool_]) -> F64:
    """Extend each tile's field from its covered cells over the whole grid, smoothly.

    A tile covers a quarter of the sphere at most; the correction must still be defined
    (and smooth) right up to the tile's border, where the blur otherwise pulls in zeros.
    Normalised convolution with a growing box: each pass fills every cell within reach of
    a covered one with the weighted mean of what it can see. `field` and `mask` carry a
    leading tile axis so all tiles are done together.
    """
    filled = np.where(mask[..., None], field, 0.0)
    covered = mask.astype(np.float64)
    radius = 2  # fine near the border first: starting at 8 cost 0.8 points of over-8 share
    while covered.min() <= 0.0 and radius <= 256:
        numerator = _box_blur(filled * covered[..., None], radius)
        denominator = _box_blur(covered[..., None], radius)[..., 0]
        fresh = (denominator > 1e-12) & (covered <= 0.0)
        filled[fresh] = numerator[fresh] / denominator[fresh][:, None]
        covered[fresh] = 1.0
        radius *= 2
    return np.asarray(filled, dtype=np.float64)
