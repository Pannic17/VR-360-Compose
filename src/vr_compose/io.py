"""Reading tiles and writing panoramas.

Two things this layer deliberately does:

* **Drops alpha on read.** Every source tile is RGBA with alpha uniformly 255, so a
  quarter of each file carries no information (AGENTS.md §4). Nothing downstream should
  ever see a fourth channel.
* **Writes RGB, not RGBA.** The reference output has alpha 253..255 -- rounding debris
  from an alpha-weighted blend. Ours has no alpha at all.
"""

from __future__ import annotations

import concurrent.futures
import pathlib
import struct
import zlib

import numpy as np
import numpy.typing as npt
from PIL import Image

from vr_compose.source import SourceSet

U8 = npt.NDArray[np.uint8]
U16 = npt.NDArray[np.uint16]
Panorama = npt.NDArray[np.uint8] | npt.NDArray[np.uint16]
"""A finished panorama: 8-bit for delivery, 16-bit for a master (P4 item 6)."""

__all__ = [
    "Panorama",
    "load_tile",
    "load_tiles",
    "write_png",
    "write_png16",
    "write_png_atomically",
]

Image.MAX_IMAGE_PIXELS = None
"""The 7680x3840 panorama trips Pillow's decompression-bomb guard."""


def load_tile(path: pathlib.Path) -> U8:
    """One tile as an ``(n, n, 3)`` uint8 array. Alpha is discarded."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def load_tiles(
    source: SourceSet, frame: int, indices: list[int], *, workers: int = 1
) -> dict[int, U8]:
    """Load the given cameras' tiles for one frame.

    `workers` above 1 uses threads: PNG decoding happens inside Pillow's C code with the
    GIL released, so threads already overlap. P2 moves to a process pool as part of the
    pipeline, where the win is larger and the frozen-executable caveat applies
    (AGENTS.md §2, constraint 4).
    """
    if workers <= 1:
        return {index: load_tile(source.tile_path(index, frame)) for index in indices}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        loaded = pool.map(lambda index: (index, load_tile(source.tile_path(index, frame))), indices)
        return dict(loaded)


def write_png16(path: pathlib.Path, image: U16, *, compress_level: int = 1) -> int:
    """Write a 16-bit RGB PNG by hand, because Pillow will not.

    `Image.fromarray` refuses three-channel uint16 outright ("Cannot handle this data
    type"), and 16-bit RGB is the one PNG variant it cannot save. The format is small
    enough to emit directly: a fixed 13-byte header, one zlib stream of scanlines each
    prefixed with a zero filter byte, and samples big-endian. `tests/conftest.py` already
    encodes PNGs this way for fixtures, so the technique is not new to this codebase.

    Scanlines use filter type 2, "Up": each byte has the byte above it subtracted, modulo
    256. That is one line of arithmetic and it is worth it here -- unfiltered, a 16-bit 8K
    master measured 153.9 MiB against the 8-bit version's 39.3, because 16-bit low bytes
    are noise-like and zlib cannot do anything with them. Vertically differencing an
    equirect attacks exactly that: adjacent rows are similar everywhere and nearly
    identical near the poles, where a row spans almost no solid angle.

    PNG defines the row above the first as all zeros, so type 2 applies uniformly with no
    special case. Correctness is checked against Pillow's own reader, which is an
    independent decoder even though it downconverts 16-bit RGB to 8.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint16:
        raise ValueError(f"expected a 16-bit RGB image, got {image.shape} {image.dtype}")
    height, width = image.shape[:2]

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    big_endian = np.ascontiguousarray(image, dtype=">u2")
    rows = big_endian.reshape(height, -1).view(np.uint8)
    raw = np.empty((height, rows.shape[1] + 1), np.uint8)
    raw[:, 0] = 2  # filter type 2 ("Up") on every scanline
    # uint8 subtraction wraps, which is exactly the modulo-256 the format specifies
    raw[0, 1:] = rows[0]
    raw[1:, 1:] = rows[1:] - rows[:-1]
    header = struct.pack(">II", width, height) + bytes([16, 2, 0, 0, 0])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw.tobytes(), compress_level))
        + chunk(b"IEND", b"")
    )
    return path.stat().st_size


def write_png_atomically(path: pathlib.Path, image: Panorama, *, compress_level: int = 6) -> int:
    """Write through `<path>.part` and rename. Returns the size in bytes.

    The master sink resumes by asking whether a frame's file exists, so a half-written
    PNG must never be able to answer yes -- the same discipline the MP4 segments use.

    Dispatches on dtype: 16-bit goes through :func:`write_png16`, 8-bit through Pillow.
    """
    partial = path.with_name(path.name + ".part")
    try:
        if image.dtype == np.uint16:
            size = write_png16(partial, np.asarray(image, np.uint16), compress_level=compress_level)
        else:
            size = write_png(partial, np.asarray(image, np.uint8), compress_level=compress_level)
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return size


def write_png(path: pathlib.Path, image: U8, *, compress_level: int = 6) -> int:
    """Write an RGB panorama and return its size in bytes.

    `compress_level` 6 is Pillow's default and costs 2.77 s at 7680x3840; level 1 costs
    0.68 s for roughly 10% more bytes. It only matters for intermediate masters -- the
    MP4 path in P2 bypasses PNG entirely.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an RGB image, got shape {image.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # `format` explicitly, not inferred from the suffix: this writes PNG whatever the file
    # is called, and the atomic path above hands it a `.png.part`, which Pillow would
    # otherwise reject as an unknown extension.
    Image.fromarray(image, mode="RGB").save(path, format="PNG", compress_level=compress_level)
    return path.stat().st_size
