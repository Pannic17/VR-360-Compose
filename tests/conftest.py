"""Shared fixtures. Everything here is synthetic: no test reads the reference data.

Tests that do need `E:\\22` are marked `needs_reference_data` and skipped unless it is
present, so the suite passes on any machine.
"""

from __future__ import annotations

import pathlib
import struct
import zlib
from collections.abc import Callable, Iterable

import numpy as np
import numpy.typing as npt
import pytest

from vr_compose import projection
from vr_compose.rig import Rig

U8 = npt.NDArray[np.uint8]

REFERENCE_ROOT = pathlib.Path("E:/22")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "needs_reference_data: requires the read-only reference set at E:/22"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if REFERENCE_ROOT.is_dir():
        return
    skip = pytest.mark.skip(reason=f"reference data not present at {REFERENCE_ROOT}")
    for item in items:
        if "needs_reference_data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def reference_root() -> pathlib.Path:
    return REFERENCE_ROOT


def encode_png(pixels: U8) -> bytes:
    """A minimal valid PNG. Avoids depending on Pillow to *produce* test fixtures."""
    height, width = pixels.shape[:2]
    channels = pixels.shape[2] if pixels.ndim == 3 else 1
    colour_type = {1: 0, 3: 2, 4: 6}[channels]

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + row.tobytes() for row in np.atleast_3d(pixels))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">II", width, height) + bytes([8, colour_type, 0, 0, 0]))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def make_source_tree(
    root: pathlib.Path,
    *,
    cameras: int,
    stem: str,
    frames: Iterable[int],
    inner: str | None = "Tempory",
    size: tuple[int, int] = (16, 16),
    digits: int = 4,
    tile_for: Callable[[int, int], U8] | None = None,
) -> pathlib.Path:
    """Write a synthetic source tree. `tile_for(camera_index, frame)` supplies pixels."""
    width, height = size
    for index in range(1, cameras + 1):
        directory = root / f"Camera{index}"
        if inner:
            directory = directory / inner
        directory.mkdir(parents=True, exist_ok=True)
        for frame in frames:
            if tile_for is None:
                pixels = np.full((height, width, 3), index * 7 % 256, np.uint8)
            else:
                pixels = tile_for(index, frame)
            (directory / f"{stem}.{frame:0{digits}d}.png").write_bytes(encode_png(pixels))
    return root


def analytic_panorama(width: int, height: int) -> U8:
    """A smooth, deliberately asymmetric panorama.

    Red rises with longitude and green with latitude, so a horizontal mirror, a vertical
    flip or a sector rotation all change the image -- which a symmetric test pattern
    would hide.
    """
    dirs = projection.equirect_directions(width, height)
    lon = np.arctan2(dirs[1], dirs[0])
    lat = np.arcsin(np.clip(dirs[2], -1.0, 1.0))
    red = (lon / (2 * np.pi) + 0.5) * 255.0
    green = (lat / np.pi + 0.5) * 255.0
    blue = (np.sin(3.0 * lon) * 0.5 + 0.5) * 255.0
    stacked = np.stack([red, green, blue], axis=-1).reshape(height, width, 3)
    return np.asarray(np.clip(stacked, 0, 255).astype(np.uint8), dtype=np.uint8)


def sample_panorama_into_tile(panorama: U8, rig: Rig, index: int, size: int) -> U8:
    """Render one rig tile by sampling `panorama`, i.e. the inverse of stitching."""
    height, width = panorama.shape[:2]
    view = rig.view_for(index)
    rays = projection.tile_rays(size, rig.fov_deg, mirrored=rig.mirrored)
    world = projection.camera_to_world(view.yaw, view.elevation) @ rays
    lon = np.arctan2(world[1], world[0])
    lat = np.arcsin(np.clip(world[2], -1.0, 1.0))
    u = np.clip(((lon / (2 * np.pi)) + 0.5) * width, 0, width - 1).astype(np.int32)
    v = np.clip((0.5 - lat / np.pi) * height, 0, height - 1).astype(np.int32)
    return np.asarray(panorama[v, u].reshape(size, size, 3), dtype=np.uint8)
