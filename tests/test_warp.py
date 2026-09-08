"""The precomputed warp must be indistinguishable from the per-frame stitch.

P1's `stitch_bands` is the verified geometry. `WarpPlan` exists purely for speed, so the
one property that matters is byte equality with it -- for any thread count, with or
without statistics, and after a save/load round trip.
"""

from __future__ import annotations

import pathlib

import numpy as np
import numpy.typing as npt
import pytest

from conftest import analytic_panorama, sample_panorama_into_tile
from vr_compose import verify
from vr_compose.rig import Rig, View, twenty_file_rig
from vr_compose.stitch import stitch_frame
from vr_compose.warp import PLAN_FORMAT, WarpPlan, plan_fingerprint

U8 = npt.NDArray[np.uint8]
WIDTH, TILE = 512, 96


@pytest.fixture(scope="module")
def rig() -> Rig:
    return twenty_file_rig()


@pytest.fixture(scope="module")
def tiles(rig: Rig) -> dict[int, U8]:
    panorama = analytic_panorama(WIDTH, WIDTH // 2)
    return {i: sample_panorama_into_tile(panorama, rig, i, TILE) for i in rig.unique_indices}


@pytest.fixture(scope="module")
def plan(rig: Rig) -> WarpPlan:
    return WarpPlan.build(rig, WIDTH, WIDTH // 2, TILE)


def test_plan_matches_stitch_byte_for_byte(rig: Rig, tiles: dict[int, U8], plan: WarpPlan) -> None:
    expected = stitch_frame(tiles, rig, WIDTH).image
    assert np.array_equal(plan.apply(tiles).image, expected)


@pytest.mark.parametrize("threads", [1, 2, 3, 7, 32])
def test_thread_count_does_not_change_a_single_byte(
    tiles: dict[int, U8], plan: WarpPlan, threads: int
) -> None:
    reference = plan.apply(tiles, threads=1).image
    assert np.array_equal(plan.apply(tiles, threads=threads).image, reference)


def test_statistics_match_the_stitch(rig: Rig, tiles: dict[int, U8], plan: WarpPlan) -> None:
    """Metric A computed through the plan must equal metric A through the stitch."""
    from_stitch = verify.agreement(stitch_frame(tiles, rig, WIDTH).stats)
    from_plan = verify.agreement(plan.apply(tiles, with_stats=True, threads=3).stats)
    assert from_plan == from_stitch


def test_stats_are_empty_unless_requested(tiles: dict[int, U8], plan: WarpPlan) -> None:
    assert plan.apply(tiles).stats.count.size == 0
    assert plan.apply(tiles, with_stats=True).stats.count.size == WIDTH * WIDTH // 2


def test_every_output_pixel_has_at_least_one_contributor(plan: WarpPlan) -> None:
    covered = np.zeros(plan.width * plan.height, bool)
    for tile in plan.tiles:
        covered[tile.out_index] = True
    assert covered.all(), "the plan must cover the whole sphere, like the rig does"


def test_contributor_lists_are_sorted_and_in_range(plan: WarpPlan) -> None:
    """Sorted output indices are what make the banded threads correct and cache-friendly."""
    for tile in plan.tiles:
        assert np.all(np.diff(tile.out_index) > 0), f"camera {tile.camera} not strictly ascending"
        assert tile.src_index.min() >= 0
        assert tile.src_index.max() < TILE * TILE
        assert np.all(tile.weight > 0)


def test_mean_contributors_matches_the_coverage_analysis(plan: WarpPlan) -> None:
    """AGENTS.md §3: 2.50 tiles per direction on average (solid-angle weighted).

    Pixel-weighted on an equirect grid over-counts the poles, so allow a loose band.
    """
    ratio = plan.entries / (plan.width * plan.height)
    assert 2.2 < ratio < 2.9, ratio


def test_save_and_load_round_trip(
    tiles: dict[int, U8], plan: WarpPlan, tmp_path: pathlib.Path
) -> None:
    target = tmp_path / "plan.npz"
    size = plan.save(target)
    assert size > 0
    loaded = WarpPlan.load(target, plan.fingerprint)
    assert loaded.fingerprint == plan.fingerprint
    assert (loaded.width, loaded.height, loaded.tile_size) == (
        plan.width,
        plan.height,
        plan.tile_size,
    )
    assert [t.camera for t in loaded.tiles] == [t.camera for t in plan.tiles]
    assert np.array_equal(loaded.apply(tiles).image, plan.apply(tiles).image)


def test_load_refuses_a_plan_for_different_inputs(plan: WarpPlan, tmp_path: pathlib.Path) -> None:
    target = tmp_path / "plan.npz"
    plan.save(target)
    with pytest.raises(ValueError, match="fingerprint"):
        WarpPlan.load(target, "0000000000000000")


def test_fingerprint_changes_with_every_input(rig: Rig) -> None:
    base = plan_fingerprint(rig, 512, 256, 96)
    assert plan_fingerprint(rig, 1024, 512, 96) != base
    assert plan_fingerprint(rig, 512, 256, 128) != base
    tilted = Rig("tilted", tuple(View(v.yaw, v.elevation + 1) for v in rig.views))
    assert plan_fingerprint(tilted, 512, 256, 96) != base
    flipped = Rig(rig.name, rig.views, rig.fov_deg, mirrored=not rig.mirrored)
    assert plan_fingerprint(flipped, 512, 256, 96) != base
    assert plan_fingerprint(rig, 512, 256, 96) == base, "and it is stable"


def test_fingerprint_embeds_the_format_version(rig: Rig) -> None:
    assert isinstance(PLAN_FORMAT, int)
    assert len(plan_fingerprint(rig, 512, 256, 96)) == 16


def test_missing_or_misshapen_tiles_are_refused(tiles: dict[int, U8], plan: WarpPlan) -> None:
    incomplete = dict(tiles)
    del incomplete[next(iter(incomplete))]
    with pytest.raises(ValueError, match="missing tile"):
        plan.apply(incomplete)
    wrong = dict(tiles)
    first = next(iter(wrong))
    wrong[first] = np.zeros((TILE + 1, TILE + 1, 3), np.uint8)
    with pytest.raises(ValueError, match="expected"):
        plan.apply(wrong)


def test_build_validates_its_inputs(rig: Rig) -> None:
    with pytest.raises(ValueError, match="2:1"):
        WarpPlan.build(rig, 100, 100, 16)
    with pytest.raises(ValueError, match="positive"):
        WarpPlan.build(rig, 64, 32, 0)


def test_extra_tiles_are_ignored(rig: Rig, tiles: dict[int, U8], plan: WarpPlan) -> None:
    """A caller may hand over all 20 files; the plan only reads the 15 it needs."""
    everything = dict(tiles)
    for index in rig.duplicate_indices:
        everything[index] = np.zeros((TILE, TILE, 3), np.uint8)
    assert np.array_equal(plan.apply(everything).image, plan.apply(tiles).image)
