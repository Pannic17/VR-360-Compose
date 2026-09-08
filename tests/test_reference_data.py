"""The independent check: real tiles, real geometry, the recorded metric A baseline.

Skipped automatically when the read-only reference set is absent, so the rest of the
suite stays machine-independent. This data is never written to (AGENTS.md §2).
"""

from __future__ import annotations

import pathlib

import pytest

from vr_compose import io, source, verify
from vr_compose.rig import rig_for
from vr_compose.stitch import stitch_frame

pytestmark = [pytest.mark.needs_reference_data, pytest.mark.slow]

FRAME = 1656
"""A frame whose tiles still exist: 1..1655 were consumed by the previous pipeline."""


@pytest.fixture(scope="module")
def reference_set(reference_root: pathlib.Path) -> source.SourceSet:
    found = [s for s in source.scan(reference_root) if s.usable]
    assert len(found) == 1, f"expected one source set in {reference_root}, got {len(found)}"
    return found[0]


def test_reference_set_matches_the_documented_shape(reference_set: source.SourceSet) -> None:
    assert reference_set.camera_count == 20
    assert reference_set.stem == "L_Cathedral"
    assert reference_set.tile_size == 1920
    assert FRAME in reference_set.frames


@pytest.mark.parametrize("sampler", ["nearest", "bilinear", "catmullrom"])
def test_metric_a_holds_on_real_tiles(reference_set: source.SourceSet, sampler: str) -> None:
    rig = rig_for(reference_set.camera_count)
    tiles = io.load_tiles(reference_set, FRAME, list(rig.unique_indices), workers=8)
    result = stitch_frame(tiles, rig, 3840, sampler=sampler)
    report = verify.agreement(result.stats)
    baseline = verify.BASELINES[sampler]

    assert report.passed, report.report(sampler)
    assert report.median == pytest.approx(baseline["median"], abs=0.05)
    assert report.mean == pytest.approx(baseline["mean"], abs=0.05)
    assert report.p95 == pytest.approx(baseline["p95"], abs=0.2)
    assert report.fraction_over_8 == pytest.approx(baseline["fraction_over_8"], abs=5e-4)
    assert report.max_contributors == 4
    assert result.overlap_fraction == pytest.approx(1.0, abs=1e-3)

    # The reference output measures 1.15/255 across the wrap; ours should be comparable.
    assert verify.wrap_seam_error(result.image) < 3.0


def test_interpolating_improves_every_metric_a_figure(reference_set: source.SourceSet) -> None:
    """The P4 claim, on real tiles: a quarter of metric A was the sampler's own error.

    Stated as an inequality rather than a number so it keeps meaning if the content
    changes -- the numbers themselves live in `verify.BASELINES`.
    """
    rig = rig_for(reference_set.camera_count)
    tiles = io.load_tiles(reference_set, FRAME, list(rig.unique_indices), workers=8)
    nearest = verify.agreement(stitch_frame(tiles, rig, 3840, sampler="nearest").stats)
    bilinear = verify.agreement(stitch_frame(tiles, rig, 3840, sampler="bilinear").stats)

    assert bilinear.median < nearest.median * 0.8
    assert bilinear.mean < nearest.mean * 0.85
    assert bilinear.p95 < nearest.p95
    assert bilinear.fraction_over_8 < nearest.fraction_over_8 * 0.25, (
        "the outliers were overwhelmingly half-pixel placement error"
    )


def test_metric_a_does_not_rank_fidelity(reference_set: source.SourceSet) -> None:
    """The trap this metric sets, pinned so nobody reads it as a quality score.

    Catmull-Rom scores *worse* on agreement than bilinear while reconstructing the source
    render 1.24 dB better (`tools/fidelity_probe.py`). Agreement rewards smoothness, and a
    sharper kernel overshoots at edges, so two tiles sampling one direction from different
    sub-pixel offsets disagree more. Both facts are real; only one of them is about detail.
    """
    rig = rig_for(reference_set.camera_count)
    tiles = io.load_tiles(reference_set, FRAME, list(rig.unique_indices), workers=8)
    bilinear = verify.agreement(stitch_frame(tiles, rig, 3840, sampler="bilinear").stats)
    cubic = verify.agreement(stitch_frame(tiles, rig, 3840, sampler="catmullrom").stats)

    assert cubic.median > bilinear.median, "if this flips, re-read verify.BASELINES"
    assert cubic.passed and bilinear.passed, "both must still clear the geometry gate"
