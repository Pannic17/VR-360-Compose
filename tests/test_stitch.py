"""Stitching, verified end-to-end against an analytic panorama.

The strongest synthetic test available: build a known panorama, render the rig's tiles
from it, stitch them back, and require the result to match. That exercises the whole
chain -- rig mapping, projection, banding, sampling, weight normalisation -- and the
panorama is deliberately asymmetric so a flip or a rotated sector cannot pass.

Frame 1656 of the reference data is the independent check, and it lives in
`test_reference_data.py`.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from conftest import analytic_panorama, sample_panorama_into_tile
from vr_compose import verify
from vr_compose.rig import Rig, View, twenty_file_rig
from vr_compose.stitch import stitch_bands, stitch_frame

U8 = npt.NDArray[np.uint8]


@pytest.fixture(scope="module")
def rig() -> Rig:
    return twenty_file_rig()


@pytest.fixture(scope="module")
def analytic_tiles(rig: Rig) -> tuple[U8, dict[int, U8]]:
    """A 1024x512 panorama and the rig's 15 tiles rendered from it at 256x256."""
    panorama = analytic_panorama(1024, 512)
    tiles = {
        index: sample_panorama_into_tile(panorama, rig, index, 256) for index in rig.unique_indices
    }
    return panorama, tiles


def test_round_trip_reproduces_the_panorama(
    rig: Rig, analytic_tiles: tuple[U8, dict[int, U8]]
) -> None:
    panorama, tiles = analytic_tiles
    result = stitch_frame(tiles, rig, 1024)
    error = np.abs(result.image.astype(np.int16) - panorama.astype(np.int16))
    # Nearest-neighbour sampling twice over, so a few grey levels of slop is expected;
    # a sign error would be off by tens.
    assert error.mean() < 3.0, f"mean error {error.mean():.2f} is too high for a round trip"
    assert float(np.median(error)) <= 1.0


def test_round_trip_would_fail_with_a_flipped_elevation(
    analytic_tiles: tuple[U8, dict[int, U8]],
) -> None:
    """Guard the guard: prove the round trip actually detects a sign error."""
    panorama, tiles = analytic_tiles
    broken = Rig(
        name="broken",
        views=tuple(View(v.yaw, -v.elevation) for v in twenty_file_rig().views),
        fov_deg=90.0,
    )
    result = stitch_frame(tiles, broken, 1024)
    error = np.abs(result.image.astype(np.int16) - panorama.astype(np.int16))
    assert error.mean() > 20.0, "a flipped elevation must not pass as a round trip"


def test_overlap_agreement_is_near_zero_for_consistent_tiles(
    rig: Rig, analytic_tiles: tuple[U8, dict[int, U8]]
) -> None:
    """Tiles rendered from one panorama agree by construction, so metric A must be tiny."""
    _, tiles = analytic_tiles
    result = stitch_frame(tiles, rig, 1024)
    report = verify.agreement(result.stats)
    assert report.passed
    assert report.median < 0.5
    assert result.overlap_fraction == pytest.approx(1.0, abs=1e-3)
    assert result.max_contributors == 4


def test_every_direction_is_covered(rig: Rig, analytic_tiles: tuple[U8, dict[int, U8]]) -> None:
    _, tiles = analytic_tiles
    result = stitch_frame(tiles, rig, 512)
    assert result.stats.count.min() >= 1, "the rig covers the whole sphere"


def test_band_size_does_not_change_the_output(
    rig: Rig, analytic_tiles: tuple[U8, dict[int, U8]]
) -> None:
    """Banding is a memory strategy; it must be bit-for-bit invisible."""
    _, tiles = analytic_tiles
    reference = stitch_frame(tiles, rig, 512, band_rows=256).image
    for band_rows in (1, 7, 64, 1000):
        assert np.array_equal(stitch_frame(tiles, rig, 512, band_rows=band_rows).image, reference)


def test_uniform_input_gives_uniform_output_and_no_seam(rig: Rig) -> None:
    """Weight normalisation: a constant sphere must come out exactly constant."""
    tiles = {i: np.full((32, 32, 3), 200, np.uint8) for i in rig.unique_indices}
    result = stitch_frame(tiles, rig, 256)
    assert np.array_equal(np.unique(result.image), np.array([200], np.uint8))
    assert verify.wrap_seam_error(result.image) == 0.0


def test_extra_tiles_are_ignored(rig: Rig) -> None:
    """A caller may hand over all 20 files; only the 15 distinct ones are needed."""
    tiles = {i: np.full((16, 16, 3), 30, np.uint8) for i in range(1, rig.file_count + 1)}
    result = stitch_frame(tiles, rig, 128)
    assert np.array_equal(np.unique(result.image), np.array([30], np.uint8))


def test_missing_tiles_are_reported_by_index(rig: Rig) -> None:
    tiles = {i: np.zeros((16, 16, 3), np.uint8) for i in rig.unique_indices}
    del tiles[rig.unique_indices[3]]
    with pytest.raises(ValueError, match="missing tiles"):
        stitch_frame(tiles, rig, 128)


def test_mixed_tile_sizes_are_refused(rig: Rig) -> None:
    tiles = {i: np.zeros((16, 16, 3), np.uint8) for i in rig.unique_indices}
    tiles[rig.unique_indices[0]] = np.zeros((32, 32, 3), np.uint8)
    with pytest.raises(ValueError, match="mixed sizes"):
        stitch_frame(tiles, rig, 128)


def test_non_square_or_non_rgb_tiles_are_refused(rig: Rig) -> None:
    rgba = {i: np.zeros((16, 16, 4), np.uint8) for i in rig.unique_indices}
    with pytest.raises(ValueError, match="square RGB tile"):
        stitch_frame(rgba, rig, 128)


@pytest.mark.parametrize(("width", "height"), [(100, 100), (100, 60)])
def test_output_must_be_two_to_one(rig: Rig, width: int, height: int) -> None:
    tiles = {i: np.zeros((16, 16, 3), np.uint8) for i in rig.unique_indices}
    with pytest.raises(ValueError, match="2:1"):
        stitch_bands(tiles, rig, width, height)


def test_odd_width_is_refused(rig: Rig) -> None:
    tiles = {i: np.zeros((16, 16, 3), np.uint8) for i in rig.unique_indices}
    with pytest.raises(ValueError, match="even"):
        stitch_frame(tiles, rig, 129)


def test_wrap_seam_error_detects_a_discontinuity() -> None:
    image = np.zeros((4, 8, 3), np.uint8)
    assert verify.wrap_seam_error(image) == 0.0
    image[:, -1] = 40
    assert verify.wrap_seam_error(image) == pytest.approx(40.0)
