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

import numpy as np
import numpy.typing as npt
from PIL import Image

from vr_compose.source import SourceSet

U8 = npt.NDArray[np.uint8]

__all__ = ["load_tile", "load_tiles", "write_png"]

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


def write_png_atomically(path: pathlib.Path, image: U8, *, compress_level: int = 6) -> int:
    """Write through `<path>.part` and rename. Returns the size in bytes.

    The master sink resumes by asking whether a frame's file exists, so a half-written
    PNG must never be able to answer yes -- the same discipline the MP4 segments use.
    """
    partial = path.with_name(path.name + ".part")
    try:
        size = write_png(partial, image, compress_level=compress_level)
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
