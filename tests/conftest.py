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
F32 = npt.NDArray[np.float32]

EXR_MAGIC = b"\x76\x2f\x31\x01"

REFERENCE_CANDIDATES = (pathlib.Path("E:/360/0904"), pathlib.Path("E:/22"))
"""Where the L_Cathedral reference set has lived. The machine's own layout is not the
project's to decide, so the first one that exists wins and the rest of the suite does
not care: it moved from `E:/22` to `E:/360/0904` on 2026-09-18."""

REFERENCE_ROOT = next(
    (path for path in REFERENCE_CANDIDATES if path.is_dir()), REFERENCE_CANDIDATES[0]
)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "needs_reference_data: requires the read-only L_Cathedral reference set "
        f"(looked for at {', '.join(str(p) for p in REFERENCE_CANDIDATES)})",
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


def levels_to_linear(levels: U8) -> F32:
    """sRGB decode of 8-bit levels: the inverse of what `vr_compose.exr` does on read."""
    x = levels.astype(np.float64) / 255.0
    linear = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    return np.asarray(linear, dtype=np.float32)


def exr_zip(raw: bytes) -> bytes:
    """OpenEXR's ZIP step: split even/odd bytes, byte predictor, deflate; raw if no gain."""
    data = np.frombuffer(raw, np.uint8)
    half = (data.size + 1) // 2
    split = np.empty_like(data)
    split[:half] = data[0::2]
    split[half:] = data[1::2]
    delta = split.astype(np.int16)
    delta[1:] = delta[1:] - split[:-1].astype(np.int16) + 128
    packed = zlib.compress((delta & 0xFF).astype(np.uint8).tobytes())
    return packed if len(packed) < len(raw) else raw


def encode_exr(linear: F32, *, compression: str = "zip", alpha: bool = True) -> bytes:
    """A minimal single-part scanline EXR: half R, G, B and (by default) a constant alpha,
    laid out as the upstream plugin writes them. An independent writer, so the reader in
    `vr_compose.exr` is not tested against itself."""
    height, width = linear.shape[:2]
    half = np.ascontiguousarray(linear, dtype="<f2")
    planes = {"R": half[..., 0], "G": half[..., 1], "B": half[..., 2]}
    if alpha:
        planes["A"] = np.ones((height, width), "<f2")
    names = sorted(planes)  # the file order the format requires
    code = {"none": 0, "zips": 2, "zip": 3}[compression]
    lines = {0: 1, 2: 1, 3: 16}[code]

    def attribute(name: str, kind: str, value: bytes) -> bytes:
        return name.encode() + b"\0" + kind.encode() + b"\0" + struct.pack("<i", len(value)) + value

    chlist = b"".join(n.encode() + b"\0" + struct.pack("<iB3xii", 1, 0, 1, 1) for n in names)
    box = struct.pack("<4i", 0, 0, width - 1, height - 1)
    header = (
        EXR_MAGIC
        + struct.pack("<i", 2)
        + attribute("channels", "chlist", chlist + b"\0")
        + attribute("compression", "compression", bytes([code]))
        + attribute("dataWindow", "box2i", box)
        + attribute("displayWindow", "box2i", box)
        + attribute("lineOrder", "lineOrder", b"\0")
        + attribute("pixelAspectRatio", "float", struct.pack("<f", 1.0))
        + attribute("screenWindowCenter", "v2f", struct.pack("<2f", 0.0, 0.0))
        + attribute("screenWindowWidth", "float", struct.pack("<f", 1.0))
        + b"\0"
    )
    chunks = []
    for first in range(0, height, lines):
        count = min(lines, height - first)
        raw = b"".join(planes[n][y].tobytes() for y in range(first, first + count) for n in names)
        payload = raw if code == 0 else exr_zip(raw)
        chunks.append(struct.pack("<ii", first, len(payload)) + payload)
    offsets = []
    position = len(header) + 8 * len(chunks)
    for chunk in chunks:
        offsets.append(position)
        position += len(chunk)
    return header + struct.pack(f"<{len(chunks)}Q", *offsets) + b"".join(chunks)


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
    name_digits: int = 1,
    tile_format: str = "png",
) -> pathlib.Path:
    """Write a synthetic source tree. `tile_for(camera_index, frame)` supplies pixels.

    `name_digits=2` names the directories `Camera01..`, as VROpPanoramic does; Camera360
    writes `Camera1..`. `tile_format="exr"` writes the same pixels as linear half EXRs,
    the way upstream's A4 output does.
    """
    width, height = size
    for index in range(1, cameras + 1):
        directory = root / f"Camera{index:0{name_digits}d}"
        if inner:
            directory = directory / inner
        directory.mkdir(parents=True, exist_ok=True)
        for frame in frames:
            if tile_for is None:
                pixels = np.full((height, width, 3), index * 7 % 256, np.uint8)
            else:
                pixels = tile_for(index, frame)
            path = directory / f"{stem}.{frame:0{digits}d}.{tile_format}"
            if tile_format == "exr":
                path.write_bytes(encode_exr(levels_to_linear(pixels)))
            else:
                path.write_bytes(encode_png(pixels))
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


def textured_panorama(width: int, height: int, *, seed: int = 0) -> U8:
    """:func:`analytic_panorama` plus an aperiodic texture, for tests that phase-correlate.

    The analytic panorama is too smooth for the layout gate (`vr_compose.layout`): its
    128 px patches hold a gradient and nothing to lock onto. The texture is a sum of
    plane waves in 3-D direction space -- continuous across the seam and the poles, with
    random wavevectors so no shift lines it up with itself again.
    """
    dirs = projection.equirect_directions(width, height)
    rng = np.random.default_rng(seed)
    field = np.zeros(dirs.shape[1])
    for _ in range(24):
        wave = rng.normal(size=3)
        wave *= rng.uniform(20.0, 150.0) / np.linalg.norm(wave)
        field += np.sin(wave @ dirs + rng.uniform(0.0, 2.0 * np.pi))
    texture = (field / np.sqrt(12.0) * 45.0).reshape(height, width, 1)
    base = analytic_panorama(width, height).astype(np.float64) * 0.6 + 50.0
    return np.asarray(np.clip(base + texture, 0, 255).astype(np.uint8), dtype=np.uint8)


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
