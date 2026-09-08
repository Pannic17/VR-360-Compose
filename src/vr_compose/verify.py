"""The geometry gate: prove a stitch is right rather than assume it.

The sphere is symmetric, so an inverted elevation or a rotated sector produces a panorama
that looks entirely plausible in a thumbnail. The only reliable check is that overlapping
tiles agree with each other, which needs no reference frame and is immune to scene motion
(AGENTS.md §9, metric A).

Reference values for the 20-file rig at 3840x1920 are in :data:`BASELINES`, one set per
sampler. They are a regression baseline: if a change moves them, either the change is an
improvement worth re-recording or it is a bug.

The gate thresholds are deliberately *not* tightened to match the better sampler: the
gate answers "is the rig upside down", where the failure signal is tens of levels, and it
has to pass for the cheap preview sampler too.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import numpy.typing as npt

from vr_compose.stitch import DEFAULT_SAMPLER, BandStats

F32 = npt.NDArray[np.float32]

__all__ = [
    "BASELINE",
    "BASELINES",
    "GATE_MAX_MEAN",
    "GATE_MAX_MEDIAN",
    "Agreement",
    "agreement",
    "wrap_seam_error",
]

BASELINES = {
    "nearest": {"median": 0.68, "mean": 1.02, "p95": 3.10, "fraction_over_8": 0.0025},
    "bilinear": {"median": 0.48, "mean": 0.77, "p95": 2.52, "fraction_over_8": 0.00037},
    "catmullrom": {"median": 0.52, "mean": 0.81, "p95": 2.58, "fraction_over_8": 0.00047},
}
"""Measured by this code on frame 1656 of the reference data, 3840x1920, per sampler.

**A quarter of what this metric used to report was our own sampler.** Switching to
bilinear (P4) moved the median from 0.68 to 0.48, the mean from 1.02 to 0.77 and the
share of directions disagreeing by more than 8 levels from 0.25% to **0.037%** -- a 6.8x
drop in the outliers. Nearest neighbour places each sample up to half a pixel from where
it belongs, and two tiles rounding a shared direction in different directions disagree by
whatever the image gradient is across that offset. What is left is closer to the floor the
source itself sets (TAA jitter, up to 1 level between tiles, AGENTS.md §3).

**This metric does not rank fidelity, and must not be read as if it did.** It measures
how closely overlapping tiles *agree*, so a blurrier reconstruction scores better merely
by being smoother, and a sharper kernel that overshoots at an edge scores worse even
where it is more faithful. Catmull-Rom sits at 0.52, *above* bilinear's 0.48, while
reconstructing the source render 1.24 dB *better* (`tools/fidelity_probe.py`). Use this
metric for what it is -- a geometry gate and a regression tripwire -- and the round-trip
probe for questions about detail.

The nearest figures are slightly under the 0.70/1.04/3.14/0.26% first recorded in
AGENTS.md §9, which came from a probe script using BT.601 luma weights; this module uses
BT.709 to match the colour space the delivery spec tags. At 7680x3840 the medians are
0.67, 0.48 and 0.52 -- mildly resolution-dependent for nearest, flat for the two
interpolating samplers, so compare like with like.
"""

BASELINE = BASELINES[DEFAULT_SAMPLER]
"""The default sampler's baseline, for callers that do not care which one ran."""

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

    def report(self, sampler: str = DEFAULT_SAMPLER) -> str:
        baseline = BASELINES.get(sampler, BASELINE)
        lines = [
            f"overlap    : {self.overlap_pixels} pixels, up to {self.max_contributors} tiles",
            f"sampler    : {sampler}",
            f"median     : {self.median:6.2f}   (baseline {baseline['median']:.2f})",
            f"mean       : {self.mean:6.2f}   (baseline {baseline['mean']:.2f})",
            f"p95        : {self.p95:6.2f}   (baseline {baseline['p95']:.2f})",
            f"std > 8    : {self.fraction_over_8:6.3%}   "
            f"(baseline {baseline['fraction_over_8']:.3%})",
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


def wrap_seam_error(image: npt.NDArray[np.unsignedinteger[Any]]) -> float:
    """Mean absolute difference between the first and last columns.

    An equirect wraps at 360 degrees, so those columns are adjacent on the sphere. The
    reference output measures 1.15/255 here; a large value means the longitude mapping is
    off by a pixel or the panorama is not actually full-circle.
    """
    # int32, not int16: a 16-bit master's samples reach 65535 and would overflow.
    left = image[:, 0].astype(np.int32)
    right = image[:, -1].astype(np.int32)
    return float(np.abs(left - right).mean())
