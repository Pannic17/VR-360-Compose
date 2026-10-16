"""The EXR reader: against an independent writer, against ffmpeg, and the layouts it refuses.
Then the EXR master writer (P15 step 3): against the reader, against ffmpeg, and its table.

`conftest.encode_exr` is the writer the reader is tested against; `vr_compose.exr` never
sees its code. ffmpeg's EXR decoder is the second opinion that shares nothing with either,
and it is the independent reading of what `exr.encode_exr` writes.
"""

from __future__ import annotations

import pathlib
import struct
import subprocess
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from conftest import analytic_panorama, encode_exr, encode_png
from vr_compose import encode, exr

F32 = npt.NDArray[np.float32]
U16 = npt.NDArray[np.uint16]

try:
    TOOLS: encode.Tools | None = encode.find_tools()
except encode.FfmpegNotFound:
    TOOLS = None
needs_ffmpeg = pytest.mark.skipif(TOOLS is None, reason="ffmpeg not available")

UPSTREAM_CAPTURE = pathlib.Path("E:/VR-DEV/Render/VROpPanoramic/A4_smoke/exr")
"""An EXR capture the upstream plugin wrote (EP1_Hil at 512, five frames), when present."""


def _linear(shape: tuple[int, int] = (40, 24), seed: int = 0) -> F32:
    """Half-representable values in 0..1 with the curve's corners in the first column."""
    rng = np.random.default_rng(seed)
    values = rng.uniform(0.0, 1.0, size=(*shape, 3)).astype(np.float16)
    values[0, 0] = [0.0, 1.0, 0.5]
    values[1, 0] = [0.0031308, 0.0005, 0.97]
    return np.asarray(values, dtype=np.float32)


def _ffmpeg_half(path: pathlib.Path, shape: tuple[int, int]) -> tuple[U16, U16]:
    """ffmpeg's half bits as `(rgb, alpha)` uint16 arrays."""
    assert TOOLS is not None
    command = [str(TOOLS.ffmpeg), "-v", "error", "-i", str(path), "-f", "rawvideo",
               "-pix_fmt", "gbrapf16le", "-"]  # fmt: skip
    raw = subprocess.run(command, capture_output=True, check=True).stdout
    g, b, r, a = np.frombuffer(raw, np.uint16).reshape(4, *shape)
    return np.stack([r, g, b], axis=-1), a


@pytest.mark.parametrize("compression", ["none", "zips", "zip"])
@pytest.mark.parametrize("alpha", [True, False])
def test_half_bits_round_trip_the_independent_writer(
    tmp_path: pathlib.Path, compression: str, alpha: bool
) -> None:
    """40 rows: two full ZIP chunks and a partial one; the alpha, when present, is skipped."""
    linear = _linear((40, 24))
    path = tmp_path / "t.exr"
    path.write_bytes(encode_exr(linear, compression=compression, alpha=alpha))
    bits = exr.load_half(path)
    assert bits.dtype == np.uint16 and bits.shape == (40, 24, 3)
    assert np.array_equal(bits, linear.astype("<f2").view(np.uint16))
    assert np.array_equal(exr.half_to_linear(bits), linear)
    header = exr.probe(path)
    assert (header.width, header.height) == (24, 40)
    assert header.channels == (("A", "B", "G", "R") if alpha else ("B", "G", "R"))
    assert header.chunk_count == (3 if compression == "zip" else 40)


def test_levels_are_the_table_applied_to_the_bits(tmp_path: pathlib.Path) -> None:
    linear = _linear()
    path = tmp_path / "t.exr"
    path.write_bytes(encode_exr(linear))
    levels = exr.load_levels(path)
    assert levels.dtype == np.float32 and levels.shape == (40, 24, 3)
    assert np.array_equal(levels, exr.srgb_levels_table()[exr.load_half(path)])
    black, white, mid = levels[0, 0].tolist()
    assert (black, white) == (0.0, 255.0)
    assert mid == pytest.approx(187.516, abs=0.01)  # sRGB(0.5) * 255


def test_the_table_is_monotone_and_clamps() -> None:
    table = exr.srgb_levels_table()
    assert table.shape == (65536,) and table.dtype == np.float32
    values = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float64)
    positive = np.isfinite(values) & (values > 0)
    ordered = np.argsort(values[positive], kind="stable")
    assert np.all(np.diff(table[positive][ordered]) >= 0)
    assert (table[np.isnan(values)] == 0.0).all()
    assert (table[values <= 0] == 0.0).all()
    assert (table[values >= 1] == 255.0).all()
    toe = np.float16(0.001)  # inside the linear segment: 12.92 x, exactly
    code = int(np.array([toe]).view(np.uint16)[0])
    assert table[code] == pytest.approx(float(toe) * 12.92 * 255.0, rel=1e-6)
    assert exr.srgb_levels_table() is table, "built once"


def _spoil_version(flag: int) -> Callable[[bytes], bytes]:
    return lambda data: data[:4] + struct.pack("<i", 2 | flag) + data[8:]


def _replace(old: bytes, new: bytes) -> Callable[[bytes], bytes]:
    def spoil(data: bytes) -> bytes:
        assert old in data
        return data.replace(old, new, 1)

    return spoil


_COMPRESSION_ZIP = b"compression\0compression\0\x01\0\0\0\x03"
_R_HALF = b"R\0\x01\0\0\0"
_DISPLAY_WINDOW = b"displayWindow\0box2i\0\x10\0\0\0" + struct.pack("<4i", 0, 0, 23, 39)


@pytest.mark.parametrize(
    ("spoil", "message"),
    [
        (_spoil_version(0x200), "tiled"),
        (_spoil_version(0x800), "deep"),
        (_spoil_version(0x1000), "multi-part"),
        (_replace(_COMPRESSION_ZIP, _COMPRESSION_ZIP[:-1] + b"\x04"), "PIZ"),
        (_replace(_COMPRESSION_ZIP, _COMPRESSION_ZIP[:-1] + b"\x08"), "DWAA"),
        (_replace(_R_HALF, b"R\0\x02\0\0\0"), "FLOAT"),
        (_replace(b"G\0", b"X\0"), "no G channel"),
        (_replace(_DISPLAY_WINDOW, _DISPLAY_WINDOW[:-8] + struct.pack("<2i", 22, 39)), "cropped"),
    ],
)
def test_layouts_the_reader_does_not_handle_are_refused_by_name(
    tmp_path: pathlib.Path, spoil: Callable[[bytes], bytes], message: str
) -> None:
    """Named at probe time, so a source set is refused before any frame is read."""
    path = tmp_path / "t.exr"
    path.write_bytes(spoil(encode_exr(_linear())))
    with pytest.raises(ValueError, match=message):
        exr.probe(path)
    with pytest.raises(ValueError, match=message):
        exr.load_half(path)


def test_not_an_exr_and_truncated_files_are_refused(tmp_path: pathlib.Path) -> None:
    png = tmp_path / "t.png"
    png.write_bytes(encode_png(np.zeros((4, 4, 3), np.uint8)))
    with pytest.raises(ValueError, match="not an EXR"):
        exr.probe(png)
    data = encode_exr(_linear())
    cut = tmp_path / "cut.exr"
    cut.write_bytes(data[: len(data) // 2])
    with pytest.raises(ValueError, match="truncated"):
        exr.load_half(cut)
    stub = tmp_path / "stub.exr"
    stub.write_bytes(data[:40])
    with pytest.raises(ValueError, match="truncated"):
        exr.probe(stub)


def test_probe_needs_only_the_header(tmp_path: pathlib.Path) -> None:
    data = encode_exr(_linear())
    header = exr.read_header(data)
    path = tmp_path / "head.exr"
    path.write_bytes(data[: header.table_offset])
    assert exr.probe(path) == header
    with pytest.raises(ValueError, match="truncated"):
        exr.load_half(path)


def test_probe_reads_more_when_the_header_is_long(tmp_path: pathlib.Path) -> None:
    """Longer than the first read: the whole file is parsed instead."""
    data = encode_exr(_linear())
    padding = b"comment\0string\0" + struct.pack("<i", 20000) + b"x" * 20000
    path = tmp_path / "long.exr"
    path.write_bytes(data[:8] + padding + data[8:])
    header = exr.probe(path)
    assert (header.width, header.height) == (24, 40)


@needs_ffmpeg
@pytest.mark.parametrize("compression", ["none", "zip"])
def test_half_bits_agree_with_ffmpeg(tmp_path: pathlib.Path, compression: str) -> None:
    linear = _linear((48, 64), seed=1)
    path = tmp_path / "t.exr"
    path.write_bytes(encode_exr(linear, compression=compression))
    rgb, alpha = _ffmpeg_half(path, (48, 64))
    assert np.array_equal(exr.load_half(path), rgb)
    assert (alpha == np.float16(1.0).view(np.uint16)).all()


@needs_ffmpeg
@pytest.mark.slow
def test_an_upstream_tile_matches_ffmpeg() -> None:
    """The format as UE actually writes it, when a capture is on this machine."""
    tile = next(UPSTREAM_CAPTURE.glob("*/Camera01/*.exr"), None)
    if tile is None:
        pytest.skip(f"no upstream EXR capture under {UPSTREAM_CAPTURE}")
    header = exr.probe(tile)
    assert header.compression == 3 and header.channels == ("A", "B", "G", "R")
    rgb, alpha = _ffmpeg_half(tile, (header.height, header.width))
    assert np.array_equal(exr.load_half(tile), rgb)
    assert (alpha == np.float16(1.0).view(np.uint16)).all()
    levels = exr.load_levels(tile)
    assert float(levels.min()) >= 0.0 and float(levels.max()) <= 255.0


# -- the master writer -----------------------------------------------------------------------


def _master_bits(width: int = 24, height: int = 40) -> U16:
    """A smooth picture as a 16-bit master would hold it, then as half bits."""
    levels = analytic_panorama(width, height).astype(np.uint16) * np.uint16(257)
    return exr.half_from_levels16(levels)


def test_the_writer_round_trips_through_the_reader_at_every_level(
    tmp_path: pathlib.Path,
) -> None:
    """Pixels identical at zlib 0, 1 and 9; 40 rows make two full ZIP chunks and a partial."""
    bits = _master_bits()
    sizes = {}
    for level in (0, 1, 9):
        path = tmp_path / f"m{level}.exr"
        sizes[level] = exr.write_exr(path, bits, compress_level=level)
        assert sizes[level] == path.stat().st_size
        assert np.array_equal(exr.load_half(path), bits), f"level {level}"
        header = exr.probe(path)
        assert header.channels == ("B", "G", "R"), "no alpha, file order"
        assert header.compression == 3 and header.chunk_count == 3
        assert (header.width, header.height) == (24, 40)
    assert sizes[0] > 40 * 24 * 6, "level 0 stores every chunk raw"
    assert sizes[9] <= sizes[1] < sizes[0]


def test_the_header_says_rec709_and_the_rest_of_the_required_set(
    tmp_path: pathlib.Path,
) -> None:
    data = exr.encode_exr(_master_bits())
    primaries = struct.pack("<8f", *exr.REC709_CHROMATICITIES)
    assert b"chromaticities\0chromaticities\0" + struct.pack("<i", 32) + primaries in data
    assert b"lineOrder\0lineOrder\0" + struct.pack("<i", 1) + b"\0" in data, "increasing y"
    for name in ("pixelAspectRatio", "screenWindowCenter", "screenWindowWidth"):
        assert name.encode() + b"\0" in data
    header = exr.read_header(data)
    assert (header.x_min, header.y_min) == (0, 0)


def test_the_writer_refuses_what_it_cannot_lay_out() -> None:
    bits = _master_bits()
    with pytest.raises(ValueError, match="half bits"):
        exr.encode_exr(np.asarray(bits[..., 0]))
    with pytest.raises(ValueError, match="half bits"):
        exr.encode_exr(np.asarray(bits, dtype=np.uint8))
    with pytest.raises(ValueError, match="compress_level"):
        exr.encode_exr(bits, compress_level=10)
    with pytest.raises(ValueError, match="16-bit"):
        exr.half_from_levels16(np.zeros((2, 2, 3), np.uint8))


@needs_ffmpeg
def test_ffmpeg_decodes_the_writers_file_to_the_bits_written(tmp_path: pathlib.Path) -> None:
    """The independent reading: ffmpeg's decoder hands back the half bits unchanged."""
    assert TOOLS is not None
    bits = _master_bits(64, 48)
    path = tmp_path / "m.exr"
    exr.write_exr(path, bits)
    command = [str(TOOLS.ffmpeg), "-v", "error", "-i", str(path), "-f", "rawvideo",
               "-pix_fmt", "gbrpf16le", "-"]  # fmt: skip
    raw = subprocess.run(command, capture_output=True, check=True).stdout
    g, b, r = np.frombuffer(raw, np.uint16).reshape(3, 48, 64)
    assert np.array_equal(np.stack([r, g, b], axis=-1), bits)


def test_the_linear_table_inverts_the_encode_within_half_precision() -> None:
    """Entry k is the half of the linear value of level k / 257: black and white exact,
    monotone, no sign or NaN, and encoding an entry back lands within 0.05 level of k / 257
    -- half's own step in the highlights, where it is coarser than a 16-bit level."""
    table = exr.linear_half_table()
    assert table.shape == (65536,) and table.dtype == np.uint16
    assert table[0] == 0 and table[65535] == np.float16(1.0).view(np.uint16)
    assert np.all(np.diff(table.astype(np.int64)) >= 0), "monotone in half's bit order"
    assert table.max() == np.float16(1.0).view(np.uint16), "nothing above 1.0, no sign bit"
    back = exr.srgb_levels_table()[table].astype(np.float64)
    levels = np.arange(65536, dtype=np.float64) / 257.0
    assert np.abs(back - levels).max() <= 0.05
    assert np.abs(back - levels)[: 3 * 257].max() <= 0.001, "finer than 16 bits below level 3"
    toe = np.float16(1.0 / 65535.0 / 12.92)  # code 1 lies on the linear segment
    assert table[1] == toe.view(np.uint16)
    assert exr.linear_half_table() is table, "built once"


def test_half_from_levels16_is_the_table_applied() -> None:
    rng = np.random.default_rng(5)
    image = rng.integers(0, 65536, size=(6, 8, 3), dtype=np.uint16)
    half = exr.half_from_levels16(image)
    assert half.dtype == np.uint16 and half.shape == image.shape
    assert np.array_equal(half, exr.linear_half_table()[image])
    assert np.array_equal(exr.half_to_linear(half) <= 1.0, np.ones_like(half, dtype=bool))
