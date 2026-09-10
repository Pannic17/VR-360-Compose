"""Arithmetic of the geometry gate itself."""

from __future__ import annotations

import numpy as np
import pytest

from vr_compose import verify
from vr_compose.stitch import BandStats


def _stats(luma_per_direction: list[list[float]]) -> BandStats:
    """Build BandStats from explicit per-direction tile luma values."""
    stats = BandStats.zeros(len(luma_per_direction))
    for i, values in enumerate(luma_per_direction):
        stats.count[i] = len(values)
        stats.luma_sum[i] = sum(values)
        stats.luma_sq_sum[i] = sum(v * v for v in values)
    return stats


def test_identical_tiles_disagree_by_zero() -> None:
    report = verify.agreement(_stats([[100.0, 100.0], [50.0, 50.0, 50.0]]))
    assert report.median == pytest.approx(0.0)
    assert report.mean == pytest.approx(0.0)
    assert report.overlap_pixels == 2
    assert report.max_contributors == 3
    assert report.passed


def test_single_tile_directions_are_excluded() -> None:
    """A direction seen by one tile has nothing to disagree with."""
    report = verify.agreement(_stats([[10.0], [10.0], [20.0, 40.0]]))
    assert report.overlap_pixels == 1
    assert report.median == pytest.approx(10.0)


def test_disagreement_is_the_population_std() -> None:
    spread = _stats([[0.0, 10.0]]).disagreement()
    assert spread.tolist() == pytest.approx([5.0])


def test_large_disagreement_fails_the_gate() -> None:
    report = verify.agreement(_stats([[0.0, 60.0]] * 10))
    assert report.median > verify.GATE_MAX_MEDIAN
    assert not report.passed
    assert "FAIL" in report.report()


def test_gate_thresholds_sit_above_the_recorded_baseline() -> None:
    assert verify.BASELINE["median"] < verify.GATE_MAX_MEDIAN
    assert verify.BASELINE["mean"] < verify.GATE_MAX_MEAN


def test_empty_stats_do_not_crash() -> None:
    report = verify.agreement(BandStats.zeros(0))
    assert report.overlap_pixels == 0
    assert report.report()


def test_band_stats_extend_concatenates() -> None:
    combined = _stats([[1.0, 3.0]]).extend(_stats([[5.0, 5.0]]))
    assert combined.count.tolist() == [2.0, 2.0]
    assert np.allclose(combined.luma_sum, [4.0, 10.0])


def test_the_gate_has_two_tiers() -> None:
    """P9: content-level disagreement warns, rig-level disagreement stops."""
    content = verify.agreement(_stats([[0.0, 2.6]] * 10))  # std 1.3: fog-sized
    assert not content.passed and not content.fatal and content.verdict == "WARN"
    rig = verify.agreement(_stats([[0.0, 12.0]] * 10))  # std 6: a mirrored rig
    assert rig.fatal and rig.verdict == "FAIL"
    fine = verify.agreement(_stats([[0.0, 1.0]] * 10))
    assert fine.passed and fine.verdict == "PASS"
    assert verify.GATE_MAX_MEDIAN < verify.GATE_FATAL_MEDIAN
    assert verify.GATE_MAX_MEAN < verify.GATE_FATAL_MEAN
    assert "WARN" in content.report() and "stop above" in content.report()
