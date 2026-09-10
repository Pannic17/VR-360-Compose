"""Photometric harmonisation: takes smooth tile differences out, leaves everything else alone.

Synthetic tiles from the analytic panorama agree by construction; adding a flat offset to
half the cameras makes them disagree the way per-view fog or exposure does. The corrected
blend must agree again, the clean blend must not move, and turning the feature off must be
the P1 arithmetic byte for byte.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from conftest import analytic_panorama, sample_panorama_into_tile
from vr_compose import harmonise, verify
from vr_compose.harmonise import Corrections, Harmoniser
from vr_compose.rig import Rig, twenty_file_rig
from vr_compose.stitch import SAMPLERS, stitch_frame
from vr_compose.warp import WarpPlan, upsample_grid

U8 = npt.NDArray[np.uint8]
WIDTH, TILE = 512, 96
OFFSET = 12


@pytest.fixture(scope="module")
def rig() -> Rig:
    return twenty_file_rig()


@pytest.fixture(scope="module")
def tiles(rig: Rig) -> dict[int, U8]:
    panorama = analytic_panorama(WIDTH, WIDTH // 2)
    return {i: sample_panorama_into_tile(panorama, rig, i, TILE) for i in rig.unique_indices}


@pytest.fixture(scope="module")
def fogged(tiles: dict[int, U8]) -> dict[int, U8]:
    """Odd cameras brighter by a flat OFFSET: what per-view fog or exposure does."""
    return {
        cam: np.clip(tile.astype(np.int16) + (OFFSET if cam % 2 else 0), 0, 255).astype(np.uint8)
        for cam, tile in tiles.items()
    }


@pytest.fixture(scope="module")
def harmoniser(rig: Rig) -> Harmoniser:
    return Harmoniser(rig, TILE)


@pytest.fixture(scope="module")
def plan(rig: Rig) -> WarpPlan:
    return WarpPlan.build(rig, WIDTH, WIDTH // 2, TILE, sampler="bilinear")


def test_the_estimate_reports_what_it_found_and_fixed(
    harmoniser: Harmoniser, fogged: dict[int, U8]
) -> None:
    found = harmoniser.estimate(fogged)
    assert isinstance(found, Corrections)
    assert set(found.grids) == set(harmoniser.plan.tiles[i].camera for i in range(15))
    for grid in found.grids.values():
        assert grid.shape == (*harmonise.GRID, 3) and grid.dtype == np.float32
        assert np.abs(grid).max() <= harmonise.CLAMP
    assert found.raw.median > 4.0, "the offset must show up as disagreement"
    assert found.corrected.median < 0.5, "and be gone after correction"
    # the peak sits at the tile border, where the fill extrapolates; bounded by the clamp
    assert OFFSET * 0.6 <= found.max_abs <= harmonise.CLAMP, found.max_abs
    assert "corrected to" in found.describe()


@pytest.mark.parametrize("sampler", list(SAMPLERS))
def test_corrected_tiles_agree_again(
    rig: Rig, harmoniser: Harmoniser, fogged: dict[int, U8], sampler: str
) -> None:
    """Metric A on the full-resolution blend, before and after, per sampler."""
    found = harmoniser.estimate(fogged)
    plan = WarpPlan.build(rig, WIDTH, WIDTH // 2, TILE, sampler=sampler)
    before = verify.agreement(plan.apply(fogged, with_stats=True).stats)
    after = verify.agreement(plan.apply(fogged, with_stats=True, corrections=found.grids).stats)
    assert before.median > 4.0 and not before.passed
    assert after.median < 0.6 and after.passed, (before, after)


def test_corrected_output_stays_between_the_two_camera_groups(
    plan: WarpPlan, harmoniser: Harmoniser, tiles: dict[int, U8], fogged: dict[int, U8]
) -> None:
    """Half the cameras were brightened by OFFSET. The correction pulls every tile towards
    the local consensus, which lies between the two groups, so the corrected image must sit
    between the clean render and the clean render plus OFFSET everywhere -- it may not
    overshoot, darken, or invent detail. (Where the two groups meet, the consensus itself
    varies with coverage, so a flat "half the offset" is not the right expectation.)"""
    found = harmoniser.estimate(fogged)
    clean = plan.apply(tiles).image.astype(np.int16)
    raw = plan.apply(fogged).image.astype(np.int16)
    fixed = plan.apply(fogged, corrections=found.grids).image.astype(np.int16)
    assert (raw - clean).max() >= OFFSET - 1, "the fixture must actually disagree"
    lift = fixed - clean
    assert lift.min() >= -2 and lift.max() <= OFFSET + 2, (lift.min(), lift.max())

    # and the blend is smoother than the uncorrected one: fewer strong seams
    def seam_energy(image: np.ndarray) -> float:
        luma = image.astype(np.float32).mean(axis=2)
        return float(np.abs(np.diff(luma, axis=1)).mean() + np.abs(np.diff(luma, axis=0)).mean())

    assert seam_energy(fixed) < seam_energy(raw)


def test_consistent_tiles_are_left_alone(
    plan: WarpPlan, harmoniser: Harmoniser, tiles: dict[int, U8]
) -> None:
    """The soft threshold removes sub-level noise; a render that agrees gets a correction
    of (almost) nothing and an output that moves by at most a couple of levels."""
    found = harmoniser.estimate(tiles)
    off = plan.apply(tiles).image.astype(np.int16)
    on = plan.apply(tiles, corrections=found.grids).image.astype(np.int16)
    assert np.abs(on - off).max() <= 2
    assert np.percentile(np.abs(on - off), 99) <= 1


@pytest.mark.parametrize("sampler", list(SAMPLERS))
def test_off_is_byte_identical_to_the_reference(
    rig: Rig, tiles: dict[int, U8], fogged: dict[int, U8], sampler: str
) -> None:
    """`corrections=None` is P1's arithmetic; the feature must not have touched it."""
    plan = WarpPlan.build(rig, WIDTH, WIDTH // 2, TILE, sampler=sampler)
    for source in (tiles, fogged):
        assert np.array_equal(
            plan.apply(source).image, stitch_frame(source, rig, WIDTH, sampler=sampler).image
        )


def test_upsample_grid_is_bilinear_and_clamped() -> None:
    """The sampler both warps transcribe: a constant grid samples to the constant, a ramp
    to the ramp, and pixels past the last cell centre hold the edge value."""
    width, height = 240, 120
    constant = np.full((60, 120, 3), 7.0, np.float32)
    out = np.arange(width * height, dtype=np.int32)
    assert np.all(upsample_grid(constant, out, width, height) == 7.0)
    ramp = np.zeros((60, 120, 3), np.float32)
    ramp[..., 0] = np.arange(120, dtype=np.float32)[None, :]
    got = upsample_grid(ramp, out, width, height)[:, 0].reshape(height, width)
    # two output pixels per cell, cell centres between them: pixel x samples x/2 - 0.25 ...
    assert got[0, 1] == pytest.approx(0.25) and got[0, 2] == pytest.approx(0.75)
    assert got[0, 3] == pytest.approx(1.25)
    # ... clamped to the first and last cell centres at the edges
    assert got[0, 0] == pytest.approx(0.0) and got[0, -1] == pytest.approx(119.0)
    assert got.dtype == np.float32


def test_missing_camera_in_corrections_is_an_error(plan: WarpPlan, tiles: dict[int, U8]) -> None:
    grids = {cam: np.zeros((*harmonise.GRID, 3), np.float32) for cam in tiles}
    del grids[next(iter(grids))]
    with pytest.raises(KeyError):
        plan.apply(tiles, corrections=grids)


def test_the_default_is_on() -> None:
    """User, 2026-09-10: the defaults must render the data sources we have."""
    assert harmonise.DEFAULT_HARMONISE is True
