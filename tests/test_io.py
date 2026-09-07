"""Tile reading and panorama writing."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from conftest import encode_png, make_source_tree
from vr_compose import io, source


def test_load_tile_discards_alpha(tmp_path: pathlib.Path) -> None:
    """Every source tile is RGBA with alpha 255; nothing downstream should see it."""
    rgba = np.zeros((4, 4, 4), np.uint8)
    rgba[..., :3] = 90
    rgba[..., 3] = 255
    path = tmp_path / "t.png"
    path.write_bytes(encode_png(rgba))
    loaded = io.load_tile(path)
    assert loaded.shape == (4, 4, 3)
    assert loaded.dtype == np.uint8
    assert (loaded == 90).all()


@pytest.mark.parametrize("workers", [1, 4])
def test_load_tiles_keys_by_camera_index(tmp_path: pathlib.Path, workers: int) -> None:
    make_source_tree(tmp_path, cameras=5, stem="S", frames=[3])
    detected = source.scan(tmp_path)[0]
    tiles = io.load_tiles(detected, 3, [1, 3, 5], workers=workers)
    assert sorted(tiles) == [1, 3, 5]
    assert all(tile.shape == (16, 16, 3) for tile in tiles.values())


def test_load_tiles_is_identical_threaded_or_not(tmp_path: pathlib.Path) -> None:
    make_source_tree(tmp_path, cameras=4, stem="S", frames=[1])
    detected = source.scan(tmp_path)[0]
    indices = [1, 2, 3, 4]
    serial = io.load_tiles(detected, 1, indices, workers=1)
    threaded = io.load_tiles(detected, 1, indices, workers=4)
    assert all(np.array_equal(serial[i], threaded[i]) for i in indices)


def test_write_png_creates_parents_and_round_trips(tmp_path: pathlib.Path) -> None:
    image = np.zeros((8, 16, 3), np.uint8)
    image[..., 0] = 255
    target = tmp_path / "deep" / "dir" / "pano.png"
    size = io.write_png(target, image)
    assert target.exists() and size > 0
    assert source.png_dimensions(target) == (16, 8)
    assert np.array_equal(io.load_tile(target), image)


def test_write_png_refuses_non_rgb(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="RGB"):
        io.write_png(tmp_path / "x.png", np.zeros((4, 4, 4), np.uint8))
