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


def test_metric_a_holds_on_real_tiles(reference_set: source.SourceSet) -> None:
    rig = rig_for(reference_set.camera_count)
    tiles = io.load_tiles(reference_set, FRAME, list(rig.unique_indices), workers=8)
    result = stitch_frame(tiles, rig, 3840)
    report = verify.agreement(result.stats)

    assert report.passed, report.report()
    assert report.median == pytest.approx(verify.BASELINE["median"], abs=0.05)
    assert report.mean == pytest.approx(verify.BASELINE["mean"], abs=0.05)
    assert report.p95 == pytest.approx(verify.BASELINE["p95"], abs=0.2)
    assert report.max_contributors == 4
    assert result.overlap_fraction == pytest.approx(1.0, abs=1e-3)

    # The reference output measures 1.15/255 across the wrap; ours should be comparable.
    assert verify.wrap_seam_error(result.image) < 3.0
