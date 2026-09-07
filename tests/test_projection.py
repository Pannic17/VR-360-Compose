"""Geometry maths. No I/O, no real data.

The sign conventions here are the ones that cost the most to get wrong: an inverted
elevation or a flipped handedness produces a panorama that looks fine on a thumbnail
because the sphere is symmetric. So each convention gets an explicit assertion.
"""

from __future__ import annotations

import numpy as np
import pytest

from vr_compose import projection


def test_half_extent_is_exactly_one_at_90_degrees() -> None:
    # tan(45 deg) == 1, which is why a 90 deg square tile is geometrically a cube face.
    assert projection.half_extent(90.0) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("fov", [0.0, -1.0, 180.0, 200.0])
def test_half_extent_rejects_impossible_fov(fov: float) -> None:
    with pytest.raises(ValueError, match="fov"):
        projection.half_extent(fov)


@pytest.mark.parametrize("angle", [0.0, 30.0, -45.0, 123.4, 359.9])
def test_rotations_are_orthonormal(angle: float) -> None:
    for matrix in (projection.rot_z(angle), projection.rot_y(angle)):
        assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(matrix) == pytest.approx(1.0, abs=1e-12)


def test_positive_elevation_looks_up() -> None:
    """The sign that a thumbnail cannot catch."""
    forward = np.array([1.0, 0.0, 0.0])
    assert (projection.camera_to_world(0.0, 45.0) @ forward)[2] > 0.5
    assert (projection.camera_to_world(0.0, -45.0) @ forward)[2] < -0.5
    assert (projection.camera_to_world(0.0, 0.0) @ forward)[2] == pytest.approx(0.0, abs=1e-12)


def test_yaw_rotates_towards_positive_y() -> None:
    rotated = projection.camera_to_world(90.0, 0.0) @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(rotated, [0.0, 1.0, 0.0], atol=1e-12)


def test_equirect_directions_are_unit_length() -> None:
    dirs = projection.equirect_directions(64, 32)
    assert dirs.shape == (3, 64 * 32)
    assert np.allclose(np.linalg.norm(dirs, axis=0), 1.0, atol=1e-12)


def test_equirect_pixel_centres_recover_their_own_longitude_and_latitude() -> None:
    """Check the mapping itself rather than raw vector components.

    A raw component test is misleading: away from the equator every component is scaled
    by cos(lat), so `x` at the left edge is nowhere near -1.
    """
    width, height = 360, 180
    dirs = projection.equirect_directions(width, height)
    lon = np.arctan2(dirs[1], dirs[0]).reshape(height, width)
    lat = np.arcsin(np.clip(dirs[2], -1.0, 1.0)).reshape(height, width)

    expected_lon = ((np.arange(width) + 0.5) / width - 0.5) * 2 * np.pi
    expected_lat = (0.5 - (np.arange(height) + 0.5) / height) * np.pi
    assert np.allclose(lon[height // 2], expected_lon, atol=1e-12)
    assert np.allclose(lat[:, 0], expected_lat, atol=1e-12)

    # Orientation, stated as assertions because a flip is otherwise invisible.
    assert lat[0, 0] > 0, "the first row is the northern hemisphere"
    assert lat[-1, 0] < 0, "the last row is the southern hemisphere"
    assert lon[0, 0] < 0 < lon[0, -1], "longitude increases left to right"


def test_equirect_row_band_matches_the_full_grid() -> None:
    """Banding is a memory strategy, so it must not change a single value."""
    width, height = 40, 20
    full = projection.equirect_directions(width, height).reshape(3, height, width)
    band = projection.equirect_directions(width, height, rows=slice(7, 13))
    assert np.array_equal(band.reshape(3, 6, width), full[:, 7:13, :])


@pytest.mark.parametrize(("width", "height"), [(1, 4), (4, 1), (0, 0)])
def test_equirect_rejects_degenerate_sizes(width: int, height: int) -> None:
    with pytest.raises(ValueError, match="at least"):
        projection.equirect_directions(width, height)


def test_tile_rays_are_unit_length_and_centred_on_forward() -> None:
    rays = projection.tile_rays(16, 90.0)
    assert rays.shape == (3, 256)
    assert np.allclose(np.linalg.norm(rays, axis=0), 1.0, atol=1e-12)
    mean = rays.mean(axis=1)
    # Mean forward component of a 90 deg square tile is ~0.794, not ~1: the corner rays
    # sit 54.7 deg off axis, which is also why the poles are under-sampled (AGENTS.md §3).
    assert mean[0] == pytest.approx(0.7937, abs=1e-3)
    assert abs(mean[1]) < 1e-12 and abs(mean[2]) < 1e-12, "a tile is symmetric about its axis"

    centre = rays.reshape(3, 16, 16)[:, 8, 8]
    assert centre[0] > 0.99, "the middle of a tile looks very nearly straight ahead"


def test_mirrored_flips_only_the_horizontal_axis() -> None:
    normal = projection.tile_rays(8, 90.0, mirrored=True)
    flipped = projection.tile_rays(8, 90.0, mirrored=False)
    assert np.allclose(normal[0], flipped[0])
    assert np.allclose(normal[1], -flipped[1])
    assert np.allclose(normal[2], flipped[2])


def test_tile_rays_rejects_empty() -> None:
    with pytest.raises(ValueError, match="positive"):
        projection.tile_rays(0, 90.0)


@pytest.mark.parametrize(("yaw", "elevation"), [(0.0, 0.0), (72.0, 45.0), (216.0, -45.0)])
@pytest.mark.parametrize("mirrored", [True, False])
def test_project_inverts_tile_rays(yaw: float, elevation: float, mirrored: bool) -> None:
    """Rays out, rays back: the round trip must recover the image-plane coordinates."""
    size, fov = 12, 90.0
    rays = projection.tile_rays(size, fov, mirrored=mirrored)
    world = projection.camera_to_world(yaw, elevation) @ rays
    x, y, visible = projection.project_to_tile(world, yaw, elevation, fov, mirrored=mirrored)
    assert visible.all(), "every ray of a tile is inside its own frustum"

    t = projection.half_extent(fov)
    grid = (np.arange(size, dtype=np.float64) + 0.5) / size * 2.0 - 1.0
    expected_x, expected_y = np.meshgrid(grid * t, grid * t)
    assert np.allclose(x, expected_x.ravel(), atol=1e-12)
    assert np.allclose(y, expected_y.ravel(), atol=1e-12)


def test_directions_behind_the_camera_are_invisible() -> None:
    behind = np.array([[-1.0], [0.0], [0.0]])
    _, _, visible = projection.project_to_tile(behind, 0.0, 0.0, 90.0)
    assert not visible.any()


def test_frustum_edge_is_inclusive_and_beyond_is_not() -> None:
    fov = 90.0
    t = projection.half_extent(fov)
    inside = np.array([[1.0], [t], [0.0]])
    outside = np.array([[1.0], [t * 1.001], [0.0]])
    assert projection.project_to_tile(inside, 0.0, 0.0, fov)[2][0]
    assert not projection.project_to_tile(outside, 0.0, 0.0, fov)[2][0]


def test_tile_pixel_index_maps_centre_and_clamps_corners() -> None:
    size, fov = 10, 90.0
    t = projection.half_extent(fov)
    centre_col, centre_row = projection.tile_pixel_index(
        np.array([0.0]), np.array([0.0]), size, fov
    )
    assert (int(centre_col[0]), int(centre_row[0])) == (size // 2, size // 2)

    edge_col, edge_row = projection.tile_pixel_index(
        np.array([-t, t]), np.array([-t, t]), size, fov
    )
    assert list(edge_col) == [0, size - 1]
    assert list(edge_row) == [0, size - 1]


def test_tile_pixel_index_is_within_bounds_for_any_input() -> None:
    size = 7
    rng = np.random.default_rng(0)
    x = rng.uniform(-5.0, 5.0, 500)
    y = rng.uniform(-5.0, 5.0, 500)
    column, row = projection.tile_pixel_index(x, y, size, 90.0)
    assert column.min() >= 0 and column.max() < size
    assert row.min() >= 0 and row.max() < size
