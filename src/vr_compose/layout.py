"""The layout gate: do two cameras that see the same direction draw it in the same place?

ROADMAP P11b. A 15-file set numbered 1..15 looks exactly like a 20-file set whose last
five directories went missing; read as the 15-file rig, the latter is mis-stitched. Only
the pixels can tell them apart, and the overlap-agreement gate (:mod:`vr_compose.verify`,
metric A) cannot do it alone: it compares pixel *values*, so an exposure or fog difference
between cameras raises it exactly as a numbering error does. This gate compares
*positions* -- each camera of a pair is warped to the equirect on its own and the overlap
is phase-correlated patch by patch, which is blind to gain and offset.

The thresholds come from measurement (AGENTS.md section 6, the P11b table): a correct
layout shifts by 0.00 deg on the reference set and 0.39-0.56 deg on the new upstream's
production captures (per-view screen-space effects, its C3); a truncated 20-file set
shifts by 1.50-2.37 deg on every mis-numbered pair.

What it does not see: the pairs are a sector's up and down cameras against its horizon
camera, so a permutation of *whole sectors* keeps every pair self-consistent and passes.
The user chose the ten pairs knowing that (2026-10-08); the truncated 20-file set, the
case this gate exists for, mis-numbers six of them.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import numpy.typing as npt

from vr_compose import projection
from vr_compose.rig import Rig

__all__ = [
    "DEFAULT_LAYOUT_GATE",
    "LAYOUT_GATES",
    "MAX_SHIFT_DEG",
    "MIN_PATCHES",
    "LayoutGateFailed",
    "LayoutReport",
    "PairShift",
    "analysis_width",
    "check",
    "pair_shifts",
    "pairs_for",
]

F64 = npt.NDArray[np.float64]
U8 = npt.NDArray[np.uint8]
Shift = tuple[float, float, float]

LAYOUT_GATES = ("on", "off")
DEFAULT_LAYOUT_GATE = "on"

PATCH = 128
"""Phase correlation window, output pixels. Big enough to hold real structure, small
enough that a position-dependent shift does not average itself away."""
STEP = PATCH // 2
MIN_CONTRAST = 6.0
"""Below this a patch is flat water or flat sky and the correlation peak is noise."""
MAX_WIDTH = 3840
"""Analysis width cap. A 1920 tile's native 7680 would quadruple the work for a shift
that is measured in tens of pixels when it matters."""
ALIGNED_PX = 0.5
"""A patch counts as aligned below this shift; reported, not judged."""

MIN_PATCHES = 5
"""A pair is *checked* with at least this many usable patches. Below it the median is a
coin toss: one patch on a dark capture read 3.07 deg where five read 0.00. The new
upstream's production captures give 7-10 per pair."""
MAX_SHIFT_DEG = 1.0
"""Any checked pair whose median shift exceeds this fails the layout. Measured: correct
layouts 0.00-0.56 deg, the truncated 20-file set 1.50-2.37 deg."""


class LayoutGateFailed(RuntimeError):
    """The layout gate failed or could not check. Stop before stitching anything."""


@dataclasses.dataclass(frozen=True, slots=True)
class PairShift:
    """One camera pair's measurement: ``(magnitude, dy, dx)`` per usable patch, in pixels."""

    a: int
    b: int
    shifts: tuple[Shift, ...]
    degrees_per_pixel: float

    @property
    def patches(self) -> int:
        return len(self.shifts)

    @property
    def checked(self) -> bool:
        return self.patches >= MIN_PATCHES

    @property
    def median_px(self) -> float:
        return float(np.median([s[0] for s in self.shifts])) if self.shifts else 0.0

    @property
    def median_deg(self) -> float:
        return self.median_px * self.degrees_per_pixel

    @property
    def aligned(self) -> float:
        """Share of patches within half a pixel."""
        if not self.shifts:
            return 0.0
        return float(np.mean([s[0] < ALIGNED_PX for s in self.shifts]))

    @property
    def failed(self) -> bool:
        return self.checked and self.median_deg > MAX_SHIFT_DEG

    def line(self) -> str:
        if not self.shifts:
            return f"C{self.a:<2d} vs C{self.b:<2d}: no patch with enough contrast in the overlap"
        dy = np.array([s[1] for s in self.shifts])
        dx = np.array([s[2] for s in self.shifts])
        return (
            f"C{self.a:<2d} vs C{self.b:<2d}: n {self.patches:3d}  median {self.median_px:5.2f} px "
            f"({self.median_deg:.2f} deg)  max {max(s[0] for s in self.shifts):5.2f}  "
            f"aligned {self.aligned * 100:5.1f}%  dy {dy.mean():+5.2f}+-{dy.std():4.2f}  "
            f"dx {dx.mean():+5.2f}+-{dx.std():4.2f}"
        )


@dataclasses.dataclass(frozen=True, slots=True)
class LayoutReport:
    pairs: tuple[PairShift, ...]
    width: int
    seconds: float

    @property
    def failing(self) -> tuple[PairShift, ...]:
        return tuple(p for p in self.pairs if p.failed)

    @property
    def unchecked(self) -> tuple[PairShift, ...]:
        return tuple(p for p in self.pairs if not p.checked)

    @property
    def verdict(self) -> str:
        """FAIL beats UNVERIFIABLE: one checked pair that disagrees is already an answer.

        Otherwise every pair must be checked -- each up and each down camera sits in
        exactly one pair, so that is what it takes for every camera to have been
        compared with a neighbour.
        """
        if self.failing:
            return "FAIL"
        if self.unchecked or not self.pairs:
            return "UNVERIFIABLE"
        return "PASS"

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"

    @property
    def worst_deg(self) -> float:
        return max((p.median_deg for p in self.pairs if p.checked), default=0.0)

    def summary(self) -> str:
        """One line for a run's report."""
        checked = sum(p.checked for p in self.pairs)
        return (
            f"layout     : {self.verdict} -- {checked} of {len(self.pairs)} camera pairs "
            f"checked, worst median shift {self.worst_deg:.2f} deg (stop above "
            f"{MAX_SHIFT_DEG:g}), {self.seconds:.1f} s"
        )

    def report(self) -> str:
        """The summary plus one line per pair, for `frame` and for a refusal."""
        return "\n".join([self.summary(), *(f"  {p.line()}" for p in self.pairs)])

    def refusal(self) -> str:
        """Why the run stops, and the way out."""
        if self.failing:
            worst = max(self.failing, key=lambda p: p.median_deg)
            reason = (
                f"cameras {worst.a} and {worst.b} draw the same directions "
                f"{worst.median_deg:.2f} deg apart ({worst.aligned:.0%} of patches aligned; "
                f"a correct layout stays under {MAX_SHIFT_DEG:g}). The directory numbering "
                "does not match the rig -- for a 15-camera set, typically a 20-camera "
                "capture with Camera16..20 missing."
            )
        else:
            names = ", ".join(f"C{p.a}/C{p.b} ({p.patches})" for p in self.unchecked)
            reason = (
                f"too little texture to compare: {names} have fewer than {MIN_PATCHES} "
                "usable patches (dark or flat overlap)."
            )
        return (
            f"layout gate {self.verdict}: {reason}\n{self.report()}\n"
            "If the cameras are known to be numbered right, run with --layout-gate off."
        )


def analysis_width(rig: Rig, tile_size: int) -> int:
    """The tile's own density, capped: 512 px tiles -> 2048, 1920 px tiles -> 3840."""
    return min(rig.native_width(tile_size), MAX_WIDTH)


def pairs_for(rig: Rig, choice: str = "all") -> list[tuple[int, int]]:
    """`all`: each sector's down and up camera against its horizon one. `ring`: down only."""
    if choice not in ("ring", "all"):
        raise ValueError(f"choice must be 'ring' or 'all', got {choice!r}")
    horizon = [i for i in rig.unique_indices if rig.view_for(i).elevation == 0.0]
    chosen = []
    for index in horizon:
        yaw = rig.view_for(index).yaw
        for elevation in (-45.0, 45.0) if choice == "all" else (-45.0,):
            partner = next(
                (
                    j
                    for j in rig.unique_indices
                    if rig.view_for(j).elevation == elevation and rig.view_for(j).yaw == yaw
                ),
                None,
            )
            if partner is not None:
                chosen.append((partner, index))
    return chosen


def _directions(width: int, rows: slice, cols: slice) -> F64:
    """:func:`projection.equirect_directions` for a rectangle of the grid."""
    height = width // 2
    row_index = np.arange(height, dtype=np.float64)[rows]
    col_index = np.arange(width, dtype=np.float64)[cols]
    lon = ((col_index + 0.5) / width - 0.5) * 2.0 * np.pi
    lat = (0.5 - (row_index + 0.5) / height) * np.pi
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    cos_lat = np.cos(lat_grid)
    dirs = np.stack([cos_lat * np.cos(lon_grid), cos_lat * np.sin(lon_grid), np.sin(lat_grid)])
    return np.asarray(dirs.reshape(3, -1), dtype=np.float64)


def warp_luma(tile: U8, rig: Rig, index: int, dirs: F64) -> tuple[F64, npt.NDArray[np.bool_]]:
    """One camera's bilinear luma at `dirs`, plus where the camera actually reaches."""
    size = tile.shape[0]
    view = rig.view_for(index)
    x, y, visible = projection.project_to_tile(
        dirs, view.yaw, view.elevation, rig.fov_deg, mirrored=rig.mirrored
    )
    px, py = projection.tile_pixel_position(x, y, size, rig.fov_deg)
    px = np.clip(np.nan_to_num(px, nan=0.0, posinf=0.0, neginf=0.0), 0, size - 2)
    py = np.clip(np.nan_to_num(py, nan=0.0, posinf=0.0, neginf=0.0), 0, size - 2)
    pixels = tile.astype(np.float64)
    luma = 0.2126 * pixels[..., 0] + 0.7152 * pixels[..., 1] + 0.0722 * pixels[..., 2]
    ix, iy = np.floor(px).astype(np.int32), np.floor(py).astype(np.int32)
    fx, fy = px - ix, py - iy
    top = luma[iy, ix] * (1 - fx) + luma[iy, ix + 1] * fx
    bottom = luma[iy + 1, ix] * (1 - fx) + luma[iy + 1, ix + 1] * fx
    return top * (1 - fy) + bottom * fy, visible


def _overlap_box(rig: Rig, a: int, b: int, width: int) -> tuple[slice, slice] | None:
    """Where the pair can share a patch, on the full grid's patch lattice.

    Found on a grid eight times coarser and padded by one coarse cell, so no shared patch
    is cut off. A box that crosses the +-180 deg seam takes the full width: patches never
    wrap, so the columns on either side are measured as the full-frame scan would.
    """
    height, coarse = width // 2, 8
    dirs = projection.equirect_directions(width // coarse, height // coarse)
    shared = np.ones(dirs.shape[1], bool)
    for index in (a, b):
        view = rig.view_for(index)
        shared &= projection.project_to_tile(
            dirs, view.yaw, view.elevation, rig.fov_deg, mirrored=rig.mirrored
        )[2]
    grid = shared.reshape(height // coarse, width // coarse)
    rows, cols = np.flatnonzero(grid.any(axis=1)), np.flatnonzero(grid.any(axis=0))
    if rows.size == 0:
        return None

    def padded(lo: int, hi: int, limit: int) -> slice:
        start = max(0, (lo - 1) * coarse // STEP * STEP)
        return slice(start, min(limit, (hi + 2) * coarse))

    wraps = cols[0] == 0 or cols[-1] == grid.shape[1] - 1 or bool(np.any(np.diff(cols) > 1))
    column_slice = slice(0, width) if wraps else padded(int(cols[0]), int(cols[-1]), width)
    return padded(int(rows[0]), int(rows[-1]), height), column_slice


def _shifts(
    left: F64, right: F64, shared: npt.NDArray[np.bool_], rows: slice, cols: slice, width: int
) -> list[Shift]:
    """Phase-correlate every usable patch of the full frame's lattice inside the box."""
    window = np.hanning(PATCH)[:, None] * np.hanning(PATCH)[None, :]
    height = width // 2
    found: list[Shift] = []
    for y in range(rows.start, height - PATCH, STEP):
        if y + PATCH > rows.stop:
            break
        for x in range(cols.start, width - PATCH, STEP):
            if x + PATCH > cols.stop:
                break
            ly, lx = y - rows.start, x - cols.start
            if not shared[ly : ly + PATCH, lx : lx + PATCH].all():
                continue
            a, b = left[ly : ly + PATCH, lx : lx + PATCH], right[ly : ly + PATCH, lx : lx + PATCH]
            if a.std() < MIN_CONTRAST:
                continue
            cross = np.fft.fft2((a - a.mean()) * window) * np.conj(
                np.fft.fft2((b - b.mean()) * window)
            )
            peak = np.fft.ifft2(cross / np.maximum(np.abs(cross), 1e-9)).real
            dy, dx = np.unravel_index(int(np.argmax(peak)), peak.shape)
            dy = dy - PATCH if dy > PATCH // 2 else dy
            dx = dx - PATCH if dx > PATCH // 2 else dx
            found.append((float(np.hypot(dx, dy)), float(dy), float(dx)))
    return found


def pair_shifts(tiles: dict[int, U8], rig: Rig, a: int, b: int, width: int) -> PairShift:
    """Measure one pair at `width` (2:1 equirect), warping only the overlap's box."""
    box = _overlap_box(rig, a, b, width)
    found: list[Shift] = []
    if box is not None:
        rows, cols = box
        shape = (rows.stop - rows.start, cols.stop - cols.start)
        dirs = _directions(width, rows, cols)
        left, seen_left = warp_luma(tiles[a], rig, a, dirs)
        right, seen_right = warp_luma(tiles[b], rig, b, dirs)
        shared = (seen_left & seen_right).reshape(shape)
        found = _shifts(left.reshape(shape), right.reshape(shape), shared, rows, cols, width)
    return PairShift(a, b, tuple(found), 360.0 / width)


def check(tiles: dict[int, U8], rig: Rig, *, width: int | None = None) -> LayoutReport:
    """The layout gate's measurement for one frame: all ten pairs at the analysis width."""
    started = time.time()
    if width is None:
        width = analysis_width(rig, next(iter(tiles.values())).shape[0])
    pairs = tuple(pair_shifts(tiles, rig, a, b, width) for a, b in pairs_for(rig, "all"))
    return LayoutReport(pairs, width, time.time() - started)
