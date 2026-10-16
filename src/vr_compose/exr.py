"""Read the EXR tiles VROpPanoramic writes, with nothing but numpy and zlib.

One layout is accepted, the one the upstream plugin's `UVROpPanoramicEXROutput` produces:
a single-part scanline image, half-float R, G and B at full resolution, NONE, ZIPS or ZIP
compression. Extra channels (the constant alpha) are skipped. Anything else -- tiles,
deep or multi-part files, PIZ and the lossy codecs, float or integer channels -- is
refused with the reason, both when a source directory is scanned (:func:`probe`) and
when a tile is read. An OpenEXR dependency would read all of those, and it is the
fallback if a DCC-written EXR ever has to be accepted; for the one writer that exists
this is a hundred lines and keeps the frozen bundle as it is (ROADMAP P15).

The values are display-referred linear: UE's film curve has been applied, the sRGB encode
has not. The PNG tiles are the same image sRGB-encoded and dithered to 8 bits, so
:func:`load_levels` encodes on read and hands back **8-bit levels as float32** -- the
quantity the rest of the pipeline already works in -- without the quantisation. A half
has 65536 codes, so the encode is a table (:func:`srgb_levels_table`), computed with
Python's `math.pow` rather than `np.power` so that the table does not depend on which
vector library numpy picked for this CPU.

Half bits read here are identical to ffmpeg's own EXR decoder's (`-pix_fmt gbrapf16le`),
which `tests/test_exr.py` uses as the independent check.

The EXR *master* (`--out-format exr`, ROADMAP P15 step 3) is written here too.
:func:`encode_exr` lays out the same single-part scanline file -- half B, G and R, no
alpha, ZIP, Rec.709 chromaticities in the header -- from a panorama given as half bits,
and :func:`half_from_levels16` makes those bits from the 16-bit stitch through a second
65536-entry table, decoding the levels back to display-referred linear. So an EXR master
is the 16-bit PNG master in another encoding, and the CPU/GPU and resume identities the
uint16 panorama already has carry over without a new path through the warp. ffmpeg
decodes the result to the bits that went in (`-pix_fmt gbrpf16le`), the same check.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import pathlib
import struct
import zlib

import numpy as np
import numpy.typing as npt

__all__ = [
    "REC709_CHROMATICITIES",
    "Header",
    "encode_exr",
    "half_from_levels16",
    "half_to_linear",
    "linear_half_table",
    "load_half",
    "load_levels",
    "probe",
    "read_header",
    "srgb_levels_table",
    "write_exr",
]

F32 = npt.NDArray[np.float32]
U8 = npt.NDArray[np.uint8]
U16 = npt.NDArray[np.uint16]

MAGIC = b"\x76\x2f\x31\x01"
LINES_PER_CHUNK = {0: 1, 2: 1, 3: 16}
"""Scanlines per chunk for the compressions read here: NONE, ZIPS and ZIP."""

_COMPRESSION_NAMES = {
    0: "NONE", 1: "RLE", 2: "ZIPS", 3: "ZIP", 4: "PIZ",
    5: "PXR24", 6: "B44", 7: "B44A", 8: "DWAA", 9: "DWAB",
}  # fmt: skip
_PIXEL_TYPES = {0: "UINT", 1: "HALF", 2: "FLOAT"}
_HALF = 1
_TILED, _DEEP, _MULTIPART = 0x200, 0x800, 0x1000
_PROBE_BYTES = 8192
_ZIP = 3

REC709_CHROMATICITIES = (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290)
"""Red, green, blue and white xy: the sRGB / Rec.709 primaries the source tiles carry."""


@dataclasses.dataclass(frozen=True, slots=True)
class Header:
    """What :func:`read_header` needs to say about a file it accepts."""

    width: int
    height: int
    x_min: int
    y_min: int
    channels: tuple[str, ...]
    """Every channel in file order; R, G and B are among them, all half."""
    compression: int
    table_offset: int
    """Where the chunk offset table starts."""

    @property
    def lines_per_chunk(self) -> int:
        return LINES_PER_CHUNK[self.compression]

    @property
    def chunk_count(self) -> int:
        return -(-self.height // self.lines_per_chunk)


class _Truncated(Exception):
    """The data ended inside the header; :func:`probe` reads more and tries again."""


def _attributes(data: bytes) -> tuple[dict[str, bytes], int]:
    """Attribute values by name, and the offset just past the header's terminator."""
    attributes: dict[str, bytes] = {}
    position = 8
    while position < len(data) and data[position] != 0:
        name_end = data.find(b"\0", position)
        type_end = data.find(b"\0", name_end + 1) if name_end >= 0 else -1
        if type_end < 0 or type_end + 5 > len(data):
            raise _Truncated
        (size,) = struct.unpack_from("<i", data, type_end + 1)
        start = type_end + 5
        if size < 0:
            raise ValueError("EXR attribute has a negative size")
        if start + size > len(data):
            raise _Truncated
        attributes[data[position:name_end].decode("ascii", "replace")] = data[start : start + size]
        position = start + size
    if position >= len(data):
        raise _Truncated
    return attributes, position + 1


def _channels(chlist: bytes) -> list[tuple[str, int, int, int]]:
    """``(name, pixel type, x sampling, y sampling)`` per channel, in file order."""
    found = []
    position = 0
    while chlist[position] != 0:
        name_end = chlist.index(b"\0", position)
        pixel_type, _linear, x_sampling, y_sampling = struct.unpack_from(
            "<iB3xii", chlist, name_end + 1
        )
        name = chlist[position:name_end].decode("ascii")
        found.append((name, pixel_type, x_sampling, y_sampling))
        position = name_end + 17
    return found


def read_header(data: bytes) -> Header:
    """Parse and validate the header at the start of `data`.

    Raises `ValueError` naming what was found for any layout this module does not read,
    and the private `_Truncated` when `data` ends inside the header.
    """
    if data[:4] != MAGIC:
        raise ValueError("not an EXR file")
    if len(data) < 8:
        raise _Truncated
    (version,) = struct.unpack_from("<i", data, 4)
    if version & 0xFF != 2:
        raise ValueError(f"EXR version {version & 0xFF} is not 2")
    for flag, what in ((_TILED, "tiled"), (_DEEP, "deep"), (_MULTIPART, "multi-part")):
        if version & flag:
            raise ValueError(f"{what} EXR files are not supported; scanline, single part only")
    attributes, table_offset = _attributes(data)
    for required in ("channels", "compression", "dataWindow", "displayWindow"):
        if required not in attributes:
            raise ValueError(f"EXR header has no {required} attribute")
    compression = attributes["compression"][0]
    if compression not in LINES_PER_CHUNK:
        name = _COMPRESSION_NAMES.get(compression, str(compression))
        raise ValueError(f"EXR compression {name} is not supported; NONE, ZIPS or ZIP only")
    x_min, y_min, x_max, y_max = struct.unpack("<4i", attributes["dataWindow"])
    if attributes["displayWindow"] != attributes["dataWindow"]:
        raise ValueError("EXR data window differs from its display window (a cropped render)")
    width, height = x_max - x_min + 1, y_max - y_min + 1
    if width < 1 or height < 1:
        raise ValueError(f"EXR data window is empty: {width}x{height}")
    try:
        channels = _channels(attributes["channels"])
    except (IndexError, struct.error, UnicodeDecodeError):
        raise ValueError("EXR channel list is malformed") from None
    names = tuple(name for name, _, _, _ in channels)
    for wanted in "RGB":
        if wanted not in names:
            raise ValueError(f"EXR has no {wanted} channel; found {list(names)}")
    for name, pixel_type, x_sampling, y_sampling in channels:
        if pixel_type != _HALF:
            kind = _PIXEL_TYPES.get(pixel_type, str(pixel_type))
            raise ValueError(f"EXR channel {name} is {kind}, not HALF")
        if x_sampling != 1 or y_sampling != 1:
            raise ValueError(f"EXR channel {name} is subsampled {x_sampling}x{y_sampling}")
    return Header(width, height, x_min, y_min, names, compression, table_offset)


def probe(path: pathlib.Path) -> Header:
    """The header alone, reading a few KB; the whole file only if the header is longer."""
    with open(path, "rb") as handle:
        head = handle.read(_PROBE_BYTES)
    try:
        return read_header(head)
    except _Truncated:
        pass
    try:
        return read_header(path.read_bytes())
    except _Truncated:
        raise ValueError("EXR header is truncated") from None


def _unzip(block: bytes, expected: int) -> U8:
    """Undo OpenEXR's ZIP: inflate, the byte predictor, then the even/odd byte split."""
    if len(block) >= expected:
        # the writer stores a chunk raw when deflate would not have made it smaller
        return np.frombuffer(block, np.uint8)
    delta = np.frombuffer(zlib.decompress(block), np.uint8).copy()
    if delta.size != expected:
        raise ValueError(f"EXR chunk inflates to {delta.size} bytes, expected {expected}")
    delta[1:] += np.uint8(128)  # the predictor's (byte - 128), modulo 256
    joined = np.cumsum(delta, dtype=np.uint8)
    half = (expected + 1) // 2
    out = np.empty(expected, np.uint8)
    out[0::2] = joined[:half]
    out[1::2] = joined[half:]
    return out


def load_half(path: pathlib.Path) -> U16:
    """The tile's R, G, B as ``(height, width, 3)`` half bits, untouched."""
    data = path.read_bytes()
    try:
        header = read_header(data)
    except _Truncated:
        raise ValueError(f"{path}: EXR header is truncated") from None
    width, height = header.width, header.height
    names = header.channels
    depth = len(names)
    lines = header.lines_per_chunk
    planes = np.empty((height, depth, width), np.uint16)
    filled = np.zeros(header.chunk_count, np.bool_)
    row_bytes = depth * width * 2
    table_end = header.table_offset + 8 * header.chunk_count
    if table_end > len(data):
        raise ValueError(f"{path}: EXR file is truncated")
    for offset in struct.unpack_from(f"<{header.chunk_count}Q", data, header.table_offset):
        if offset < table_end or offset + 8 > len(data):
            raise ValueError(f"{path}: EXR chunk offset {offset} is outside the file")
        y, size = struct.unpack_from("<ii", data, offset)
        first = y - header.y_min
        if not 0 <= first < height or first % lines:
            raise ValueError(f"{path}: EXR chunk starts at row {y}, outside the image")
        count = min(lines, height - first)
        block = data[offset + 8 : offset + 8 + size]
        if len(block) != size:
            raise ValueError(f"{path}: EXR file is truncated")
        expected = count * row_bytes
        raw = _unzip(block, expected) if header.compression else np.frombuffer(block, np.uint8)
        if raw.size != expected:
            raise ValueError(f"{path}: EXR chunk holds {raw.size} bytes, expected {expected}")
        planes[first : first + count] = raw.view("<u2").reshape(count, depth, width)
        filled[first // lines] = True
    if not filled.all():
        raise ValueError(f"{path}: EXR file is missing {int((~filled).sum())} chunk(s)")
    rgb = [names.index(channel) for channel in "RGB"]
    return np.ascontiguousarray(np.stack([planes[:, k, :] for k in rgb], axis=-1))


def half_to_linear(bits: U16) -> F32:
    """The display-referred linear values a half array holds."""
    return np.asarray(bits.view(np.float16), dtype=np.float32)


def _srgb_level(linear: float) -> float:
    """IEC 61966-2-1 encode of one value clamped to 0..1, as an 8-bit level."""
    if math.isnan(linear) or linear <= 0.0:
        return 0.0
    if linear >= 1.0:
        return 255.0
    toe = linear <= 0.0031308
    encoded = linear * 12.92 if toe else 1.055 * math.pow(linear, 1.0 / 2.4) - 0.055
    return encoded * 255.0


@functools.cache
def srgb_levels_table() -> F32:
    """The sRGB-encoded 8-bit level of every half code, as float32. 65536 entries."""
    linear = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float64)
    table = np.empty(65536, np.float32)
    for code, value in enumerate(linear.tolist()):
        table[code] = _srgb_level(value)
    return table


def load_levels(path: pathlib.Path) -> F32:
    """The tile as ``(height, width, 3)`` float32 sRGB levels on the 0..255 scale."""
    return np.asarray(srgb_levels_table()[load_half(path)], dtype=np.float32)


def _linear(x: float) -> float:
    """IEC 61966-2-1 decode of one sRGB-encoded value in 0..1."""
    return x / 12.92 if x <= 0.04045 else math.pow((x + 0.055) / 1.055, 2.4)


@functools.cache
def linear_half_table() -> U16:
    """The half bits of the linear value of every 16-bit master code. 65536 entries.

    A 16-bit master code is a level times 257 (:data:`vr_compose.stitch.SIXTEEN_BIT_SCALE`),
    so entry `k` decodes `k / 65535`. The decode of :func:`srgb_levels_table`'s encode,
    built the same way -- `math.pow` -- and rounded to half by numpy's float64 conversion,
    which is round-to-nearest-even on every CPU. Half is coarser than a 16-bit level above
    level 3 (0.08 level in the highlights, which is the source EXR's own precision) and
    finer below it; encoding an entry back lands within 0.05 level of its code everywhere
    (`tests/test_exr.py`).
    """
    linear = np.array([_linear(code / 65535.0) for code in range(65536)], np.float64)
    return np.asarray(linear.astype(np.float16).view(np.uint16), dtype=np.uint16)


def half_from_levels16(image: U16) -> U16:
    """A 16-bit panorama as the half bits of its linear values, same shape."""
    if image.dtype != np.uint16:
        raise ValueError(f"expected a 16-bit panorama, got {image.dtype}")
    return np.asarray(linear_half_table()[image], dtype=np.uint16)


def _attribute(name: str, kind: str, value: bytes) -> bytes:
    return name.encode() + b"\0" + kind.encode() + b"\0" + struct.pack("<i", len(value)) + value


def _zip(raw: U8, compress_level: int) -> bytes:
    """OpenEXR's ZIP of one chunk, the inverse of :func:`_unzip`: the even/odd byte split,
    the byte predictor, deflate. Stored raw when deflate would not have made it smaller."""
    half = (raw.size + 1) // 2
    split = np.empty_like(raw)
    split[:half] = raw[0::2]
    split[half:] = raw[1::2]
    delta = split.copy()
    delta[1:] -= split[:-1]  # uint8 wraps: the predictor's modulo 256
    delta[1:] += np.uint8(128)
    packed = zlib.compress(delta.tobytes(), compress_level)
    return packed if len(packed) < raw.size else raw.tobytes()


def encode_exr(half: U16, *, compress_level: int = 1) -> bytes:
    """An EXR of a panorama given as ``(height, width, 3)`` half bits, R, G, B.

    Single part, scanline, half B, G and R, no alpha, ZIP (16 rows a chunk), line order
    increasing, data window equal to display window, Rec.709 chromaticities. The chunks
    are deflated at `compress_level`; every level decodes to the same bits, and a chunk
    deflate does not shrink is stored raw, as the format allows -- level 0 stores them all.
    """
    if half.ndim != 3 or half.shape[2] != 3 or half.dtype != np.uint16:
        raise ValueError(f"expected (height, width, 3) half bits, got {half.shape} {half.dtype}")
    if not 0 <= compress_level <= 9:
        raise ValueError(f"compress_level must be 0..9, got {compress_level}")
    height, width = half.shape[:2]
    channels = b"".join(
        name.encode() + b"\0" + struct.pack("<iB3xii", _HALF, 0, 1, 1) for name in "BGR"
    )
    box = struct.pack("<4i", 0, 0, width - 1, height - 1)
    primaries = struct.pack("<8f", *REC709_CHROMATICITIES)
    header = (
        MAGIC
        + struct.pack("<i", 2)
        + _attribute("channels", "chlist", channels + b"\0")
        + _attribute("chromaticities", "chromaticities", primaries)
        + _attribute("compression", "compression", bytes([_ZIP]))
        + _attribute("dataWindow", "box2i", box)
        + _attribute("displayWindow", "box2i", box)
        + _attribute("lineOrder", "lineOrder", b"\0")
        + _attribute("pixelAspectRatio", "float", struct.pack("<f", 1.0))
        + _attribute("screenWindowCenter", "v2f", struct.pack("<2f", 0.0, 0.0))
        + _attribute("screenWindowWidth", "float", struct.pack("<f", 1.0))
        + b"\0"
    )
    # each scanline holds its B samples, then G, then R, as little-endian halves
    planes = np.ascontiguousarray(half[:, :, ::-1].transpose(0, 2, 1), dtype="<u2")
    rows = planes.reshape(height, -1).view(np.uint8)
    lines = LINES_PER_CHUNK[_ZIP]
    chunks: list[bytes] = []
    for first in range(0, height, lines):
        payload = _zip(rows[first : first + lines].reshape(-1), compress_level)
        chunks.append(struct.pack("<ii", first, len(payload)) + payload)
    offsets: list[int] = []
    position = len(header) + 8 * len(chunks)
    for chunk in chunks:
        offsets.append(position)
        position += len(chunk)
    return header + struct.pack(f"<{len(chunks)}Q", *offsets) + b"".join(chunks)


def write_exr(path: pathlib.Path, half: U16, *, compress_level: int = 1) -> int:
    """Write :func:`encode_exr`'s bytes to `path` and return the size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = encode_exr(half, compress_level=compress_level)
    path.write_bytes(data)
    return len(data)
