"""The geometry gate: prove a stitch is right rather than assume it.

The sphere is symmetric, so an inverted elevation or a rotated sector produces a panorama
that looks entirely plausible in a thumbnail. The only reliable check is that overlapping
tiles agree with each other, which needs no reference frame and is immune to scene motion
(AGENTS.md §9, metric A).

Reference values for the 20-file rig with nearest-neighbour sampling at 3840x1920 are in
:data:`BASELINE`. They are a regression baseline: if a change moves them, either the
change is an improvement worth re-recording or it is a bug.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import numpy.typing as npt

from vr_compose.stitch import BandStats

F32 = npt.NDArray[np.float32]

__all__ = [
    "BASELINE",
    "GATE_MAX_MEAN",
    "GATE_MAX_MEDIAN",
    "Agreement",
    "agreement",
    "wrap_seam_error",
]

BASELINE = {"median": 0.68, "mean": 1.02, "p95": 3.10, "fraction_over_8": 0.0025}
"""Measured by this code on frame 1656 of the reference data, 3840x1920, nearest neighbour.

Slightly under the 0.70/1.04/3.14/0.26% first recorded in AGENTS.md §9, which came from a
probe script using BT.601 luma weights; this module uses BT.709 to match the colour space
the delivery spec tags. At 7680x3840 the median is 0.67 -- the metric is mildly
resolution-dependent, so compare like with like.
"""

GATE_MAX_MEDIAN = 1.0
GATE_MAX_MEAN = 1.5
"""Pass thresholds. Comfortably above the baseline, far below a sign-flip's error."""


@dataclasses.dataclass(frozen=True, slots=True)
class Agreement:
    """How closely overlapping tiles agree, on a 0-255 luma scale."""

    median: float
    mean: float
    p95: float
    fraction_over_8: float
    overlap_pixels: int
    max_contributors: int

    @property
    def passed(self) -> bool:
        return self.median <= GATE_MAX_MEDIAN and self.mean <= GATE_MAX_MEAN

    def report(self) -> str:
        lines = [
            f"overlap    : {self.overlap_pixels} pixels, up to {self.max_contributors} tiles",
            f"median     : {self.median:6.2f}   (baseline {BASELINE['median']:.2f})",
            f"mean       : {self.mean:6.2f}   (baseline {BASELINE['mean']:.2f})",
            f"p95        : {self.p95:6.2f}   (baseline {BASELINE['p95']:.2f})",
            f"std > 8    : {self.fraction_over_8:6.2%}   "
            f"(baseline {BASELINE['fraction_over_8']:.2%})",
            f"verdict    : {'PASS' if self.passed else 'FAIL'}   "
            f"(gate: median <= {GATE_MAX_MEDIAN}, mean <= {GATE_MAX_MEAN})",
        ]
        return "\n".join(lines)


def agreement(stats: BandStats) -> Agreement:
    """Summarise per-direction tile disagreement into the gate metric."""
    spread = stats.disagreement()
    if spread.size == 0:
        return Agreement(0.0, 0.0, 0.0, 0.0, 0, int(stats.count.max()) if stats.count.size else 0)
    return Agreement(
        median=float(np.median(spread)),
        mean=float(spread.mean()),
        p95=float(np.percentile(spread, 95)),
        fraction_over_8=float((spread > 8.0).mean()),
        overlap_pixels=int(spread.size),
        max_contributors=int(stats.count.max()),
    )


def wrap_seam_error(image: npt.NDArray[np.uint8]) -> float:
    """Mean absolute difference between the first and last columns.

    An equirect wraps at 360 degrees, so those columns are adjacent on the sphere. The
    reference output measures 1.15/255 here; a large value means the longitude mapping is
    off by a pixel or the panorama is not actually full-circle.
    """
    left = image[:, 0].astype(np.int16)
    right = image[:, -1].astype(np.int16)
    return float(np.abs(left - right).mean())
