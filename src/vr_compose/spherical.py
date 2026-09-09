"""Spherical Video V2 metadata (`st3d` / `sv3d`), written into a finished MP4 by hand.

A 360 file that carries no projection metadata is just a very wide video: players show
it flat, or letterboxed, and the delivery is useless without the viewer being told what
it is looking at. So this is a conformance requirement, not a nicety.

**Why by hand.** ffmpeg cannot write it -- its spherical support is demuxer-side only, so
the metadata survives a transcode but cannot be created. The PyPI packages that look like
they would do it (`spatialmedia`, `spatial-media`) do not exist, verified. Spherical Video
V2 itself is a handful of fixed-size boxes and has not changed since 2017, so writing it
here costs less than a dependency would and can be read back and asserted byte for byte.

**Where it goes.** Inside the *visual sample entry* (`avc1` / `hvc1`), as a sibling of
`avcC` / `hvcC`::

    moov/trak/mdia/minf/stbl/stsd/avc1
        st3d                    stereo layout (0 = monoscopic)
        sv3d
            svhd                who wrote it
            proj
                prhd            pose: yaw / pitch / roll
                equi            equirectangular, with its bounds

**The part that is easy to get wrong.** Growing `moov` moves everything after it, and
`stco` / `co64` hold *absolute file offsets* into `mdat`. Our files are written with
`+faststart`, i.e. `moov` first, so every chunk offset shifts by exactly the number of
bytes inserted. Miss that and the file still parses, still reports the metadata, still
shows a duration -- and decodes to garbage or nothing at all, because every chunk pointer
is short by ~100 bytes. :func:`inject` therefore patches every chunk offset that points
past the insertion point, which is also correct for the `mdat`-first layout (nothing
points past it, so nothing is patched).
"""

from __future__ import annotations

import dataclasses
import pathlib
import struct
from collections.abc import Iterator
from typing import BinaryIO

__all__ = ["MONOSCOPIC", "STEREO_LEFT_RIGHT", "STEREO_TOP_BOTTOM", "Spherical", "inject", "read"]

MONOSCOPIC = 0
STEREO_TOP_BOTTOM = 1
STEREO_LEFT_RIGHT = 2
_LAYOUTS = {
    MONOSCOPIC: "monoscopic",
    STEREO_TOP_BOTTOM: "top-bottom",
    STEREO_LEFT_RIGHT: "left-right",
}

_CONTAINERS = frozenset(
    {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"mvex", b"moof", b"traf"}
)
_SAMPLE_ENTRIES = frozenset({b"avc1", b"avc3", b"hvc1", b"hev1", b"av01", b"vp09"})
_VISUAL_SAMPLE_ENTRY_PAYLOAD = 78
"""Fixed part of a VisualSampleEntry before its extension boxes begin (ISO/IEC 14496-12).

6 reserved + 2 data_reference_index + 16 pre_defined/reserved + 2 width + 2 height
+ 4 horizresolution + 4 vertresolution + 4 reserved + 2 frame_count + 32 compressorname
+ 2 depth + 2 pre_defined = 78.
"""


class SphericalError(RuntimeError):
    """The file is not a plain MP4 this module can annotate."""


@dataclasses.dataclass(frozen=True, slots=True)
class Spherical:
    """What the boxes say. Defaults describe our delivery: mono, full-frame equirect."""

    stereo_mode: int = MONOSCOPIC
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    metadata_source: str = "VR-Compose"
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    """Fraction of the frame cropped from top / bottom / left / right. All zero = the
    frame covers the whole sphere, which is what the stitcher produces."""

    def __post_init__(self) -> None:
        if self.stereo_mode not in (MONOSCOPIC, STEREO_TOP_BOTTOM, STEREO_LEFT_RIGHT):
            raise ValueError(f"stereo_mode must be 0, 1 or 2, got {self.stereo_mode}")
        if any(not 0.0 <= b < 1.0 for b in self.bounds):
            raise ValueError(f"projection bounds must each be in [0, 1), got {self.bounds}")
        if self.bounds[0] + self.bounds[1] >= 1.0 or self.bounds[2] + self.bounds[3] >= 1.0:
            raise ValueError(f"projection bounds leave no picture: {self.bounds}")

    def boxes(self) -> bytes:
        """`st3d` followed by `sv3d`, ready to append to a visual sample entry."""
        st3d = _full_box(b"st3d", bytes([self.stereo_mode]))
        svhd = _full_box(b"svhd", self.metadata_source.encode("utf-8") + b"\x00")
        pose = (self.yaw_deg, self.pitch_deg, self.roll_deg)
        prhd = _full_box(b"prhd", struct.pack(">iii", *(_fixed_16_16(a) for a in pose)))
        equi = _full_box(b"equi", struct.pack(">4I", *(_fixed_0_32(b) for b in self.bounds)))
        return st3d + _box(b"sv3d", svhd + _box(b"proj", prhd + equi))

    def describe(self) -> str:
        pose = f"yaw {self.yaw_deg:g} pitch {self.pitch_deg:g} roll {self.roll_deg:g}"
        layout = _LAYOUTS[self.stereo_mode]
        crop = "" if not any(self.bounds) else f", bounds {self.bounds}"
        return f"equirectangular, {layout}, {pose}, source {self.metadata_source!r}{crop}"


def _box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + kind + payload


def _full_box(kind: bytes, payload: bytes, version: int = 0, flags: int = 0) -> bytes:
    return _box(kind, bytes([version]) + flags.to_bytes(3, "big") + payload)


def _fixed_16_16(degrees: float) -> int:
    return round(degrees * 65536.0)


def _fixed_0_32(fraction: float) -> int:
    """0.32 unsigned fixed point. 1.0 is not representable, hence the clamp below it."""
    return min(round(fraction * 4294967296.0), 0xFFFFFFFF)


@dataclasses.dataclass(frozen=True, slots=True)
class _Box:
    kind: bytes
    start: int
    """Offset of the box header."""
    payload: int
    """Offset of the first byte after the header (past a 64-bit size, if there is one)."""
    end: int
    size_offset: int
    size_width: int
    """4 for a 32-bit size, 8 for the `largesize` form, 0 for a box that runs to EOF."""


def _walk(buf: bytes, start: int, end: int) -> Iterator[_Box]:
    pos = start
    while pos + 8 <= end:
        size = int.from_bytes(buf[pos : pos + 4], "big")
        kind = buf[pos + 4 : pos + 8]
        payload, size_offset, size_width = pos + 8, pos, 4
        if size == 1:
            size = int.from_bytes(buf[pos + 8 : pos + 16], "big")
            payload, size_offset, size_width = pos + 16, pos + 8, 8
        elif size == 0:
            # "to end of file", legal only for the last box; it needs no size fix-up.
            size, size_width = end - pos, 0
        if size < payload - pos or pos + size > end:
            raise SphericalError(f"box {kind!r} at {pos} claims {size} bytes, past its parent")
        yield _Box(kind, pos, payload, pos + size, size_offset, size_width)
        pos += size


def _child(buf: bytes, parent: _Box, kind: bytes) -> _Box:
    for box in _walk(buf, parent.payload, parent.end):
        if box.kind == kind:
            return box
    raise SphericalError(f"no {kind.decode()} box inside {parent.kind.decode()}")


def _handler(buf: bytes, mdia: _Box) -> bytes:
    """`vide` for the video track. Past version/flags *and* the `pre_defined` word."""
    hdlr = _child(buf, mdia, b"hdlr")
    return buf[hdlr.payload + 8 : hdlr.payload + 12]


def _video_track(buf: bytes, moov: _Box) -> _Box:
    tracks = [box for box in _walk(buf, moov.payload, moov.end) if box.kind == b"trak"]
    video = [t for t in tracks if _handler(buf, _child(buf, t, b"mdia")) == b"vide"]
    if not video:
        raise SphericalError(f"no video track among {len(tracks)} track(s)")
    if len(video) > 1:
        raise SphericalError(
            f"{len(video)} video tracks; which one is the panorama is not ours to guess"
        )
    return video[0]


def _sample_entry(buf: bytes, trak: _Box) -> tuple[_Box, list[_Box]]:
    """The visual sample entry, and the chain of boxes whose sizes contain it."""
    mdia = _child(buf, trak, b"mdia")
    minf = _child(buf, mdia, b"minf")
    stbl = _child(buf, minf, b"stbl")
    stsd = _child(buf, stbl, b"stsd")
    # stsd is a FullBox with a 4-byte entry_count before the entries themselves.
    for box in _walk(buf, stsd.payload + 8, stsd.end):
        if box.kind in _SAMPLE_ENTRIES:
            return box, [trak, mdia, minf, stbl, stsd, box]
    raise SphericalError("no recognised visual sample entry in stsd")


def _chunk_offset_tables(buf: bytes, moov: _Box) -> list[_Box]:
    """Every `stco` / `co64` in the file, found by path -- never by scanning for bytes.

    A byte scan would eventually match inside a codec's parameter sets and corrupt a
    file in a way that only shows up on playback.
    """
    tables: list[_Box] = []
    for trak in _walk(buf, moov.payload, moov.end):
        if trak.kind != b"trak":
            continue
        stbl = _child(buf, _child(buf, _child(buf, trak, b"mdia"), b"minf"), b"stbl")
        for box in _walk(buf, stbl.payload, stbl.end):
            if box.kind in (b"stco", b"co64"):
                tables.append(box)
    return tables


def _shift_chunk_offsets(data: bytearray, table: _Box, insert_at: int, delta: int) -> int:
    """Add `delta` to offsets pointing past `insert_at`. Returns how many were moved."""
    wide = table.kind == b"co64"
    width = 8 if wide else 4
    count = int.from_bytes(data[table.payload + 4 : table.payload + 8], "big")
    first = table.payload + 8
    if first + count * width > table.end:
        raise SphericalError(f"{table.kind.decode()} claims {count} entries but is too short")
    moved = 0
    for i in range(count):
        at = first + i * width
        value = int.from_bytes(data[at : at + width], "big")
        if value < insert_at:
            continue
        new = value + delta
        if not wide and new > 0xFFFFFFFF:
            raise SphericalError("a 32-bit chunk offset would overflow; the file needs co64")
        data[at : at + width] = new.to_bytes(width, "big")
        moved += 1
    return moved


COPY_CHUNK = 8 << 20


def _top_level_by_seeking(handle: BinaryIO, size: int) -> list[_Box]:
    """The top-level boxes, read 16 bytes at a time instead of loading the file.

    A delivery MP4 is about 2 GB (AGENTS.md section 2, constraint 7: nothing here reads a
    whole sequence into memory). Only `moov` is ever needed in full, and it is a few
    kilobytes, so the file is walked by seeking and `mdat` is never read at all.
    """
    boxes: list[_Box] = []
    pos = 0
    while pos + 8 <= size:
        handle.seek(pos)
        header = handle.read(16)
        if len(header) < 8:
            break
        length = int.from_bytes(header[:4], "big")
        kind = header[4:8]
        payload, size_offset, size_width = pos + 8, pos, 4
        if length == 1:
            length = int.from_bytes(header[8:16], "big")
            payload, size_offset, size_width = pos + 16, pos + 8, 8
        elif length == 0:
            length, size_width = size - pos, 0
        if length < payload - pos or pos + length > size:
            raise SphericalError(f"box {kind!r} at {pos} claims {length} bytes, past the file")
        boxes.append(_Box(kind, pos, payload, pos + length, size_offset, size_width))
        pos += length
    return boxes


def _rebased(box: _Box) -> _Box:
    """The same box, addressed from its own first byte -- the coordinates of its buffer.

    Every offset shifts by the box's own start rather than being written out literally,
    so a `moov` that used the 64-bit `largesize` header keeps its 16-byte payload offset
    instead of being told it has an 8-byte one.
    """
    return dataclasses.replace(
        box,
        start=0,
        payload=box.payload - box.start,
        end=box.end - box.start,
        size_offset=box.size_offset - box.start,
    )


def _read_moov(path: pathlib.Path) -> tuple[bytes, _Box, int]:
    """`(moov bytes, the moov box in *file* coordinates, file size)`.

    The returned buffer starts at the `moov` header, so offsets inside it are
    moov-relative; only the chunk offset tables hold file-absolute numbers, and those are
    the one place the two coordinate systems have to be reconciled.
    """
    size = path.stat().st_size
    with path.open("rb") as handle:
        boxes = _top_level_by_seeking(handle, size)
        if any(box.kind == b"moof" for box in boxes):
            raise SphericalError("fragmented MP4; sample entries are not the whole story there")
        moov = next((box for box in boxes if box.kind == b"moov"), None)
        if moov is None:
            raise SphericalError("no top-level moov box; is this an MP4?")
        handle.seek(moov.start)
        buf = handle.read(moov.end - moov.start)
    return buf, moov, size


def inject(path: pathlib.Path, spherical: Spherical | None = None) -> int:
    """Write the metadata into `path`, in place. Returns the number of bytes added.

    Idempotent by refusal, not by overwrite: a file that already carries `sv3d` is left
    exactly as it is and 0 is returned, so a resumed or re-run job cannot end up with the
    boxes twice.

    The rewrite goes to a temporary file next to the target and is renamed over it, so an
    interrupted run leaves either the original or the finished file, never a half-written
    MP4 where a valid one used to be. Only `moov` is held in memory; the rest of the file
    is copied through in chunks.
    """
    spherical = spherical or Spherical()
    buf, moov_in_file, size = _read_moov(path)
    moov = _rebased(moov_in_file)
    entry, chain = _sample_entry(buf, _video_track(buf, moov))
    extensions = _walk(buf, entry.payload + _VISUAL_SAMPLE_ENTRY_PAYLOAD, entry.end)
    if any(box.kind in (b"sv3d", b"st3d") for box in extensions):
        return 0

    payload = spherical.boxes()
    delta = len(payload)
    insert_at = entry.end  # moov-relative, like everything else parsed out of `buf`
    out = bytearray(buf)
    # Chunk offsets are patched *before* the splice, while every offset from the walk
    # above is still valid. Doing it afterwards would need each table's own position
    # corrected by delta as well -- the same fix-up, applied twice, in two directions.
    # They are file-absolute, so they are compared against the absolute insertion point.
    for table in _chunk_offset_tables(buf, moov):
        _shift_chunk_offsets(out, table, moov_in_file.start + insert_at, delta)
    out[insert_at:insert_at] = payload
    # `moov` included: it is the outermost box that now holds more bytes than it says.
    for box in [moov, *chain]:
        if box.size_width == 0:
            raise SphericalError(f"{box.kind.decode()} runs to EOF; its size cannot be grown")
        current = int.from_bytes(out[box.size_offset : box.size_offset + box.size_width], "big")
        grown = current + delta
        if box.size_width == 4 and grown > 0xFFFFFFFF:
            raise SphericalError(f"{box.kind.decode()} would exceed a 32-bit size")
        out[box.size_offset : box.size_offset + box.size_width] = grown.to_bytes(
            box.size_width, "big"
        )

    temporary = path.with_name(path.name + ".sv3d")
    try:
        with path.open("rb") as source, temporary.open("wb") as target:
            _copy(source, target, 0, moov_in_file.start)
            target.write(bytes(out))
            _copy(source, target, moov_in_file.end, size - moov_in_file.end)
    except BaseException:
        # A full disk or a Ctrl-C here must not leave a stray half-file beside a finished
        # delivery, the same contract the segment writer and the concat keep.
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(path)
    return delta


def _copy(source: BinaryIO, target: BinaryIO, start: int, length: int) -> None:
    source.seek(start)
    while length > 0:
        block = source.read(min(COPY_CHUNK, length))
        if not block:
            raise SphericalError("the file ended early while being rewritten")
        target.write(block)
        length -= len(block)


def read(path: pathlib.Path) -> Spherical | None:
    """Parse back what :func:`inject` wrote, or `None` if the file carries no `sv3d`.

    Deliberately independent of the writer: it walks the file the way a player would, so
    a test asserting `read(inject(x))` catches a box that was written into the wrong
    parent or with a stale size.
    """
    buf, in_file, _size = _read_moov(path)
    moov = _rebased(in_file)
    entry, _chain = _sample_entry(buf, _video_track(buf, moov))
    children = {
        box.kind: box for box in _walk(buf, entry.payload + _VISUAL_SAMPLE_ENTRY_PAYLOAD, entry.end)
    }
    if b"sv3d" not in children:
        return None
    sv3d = children[b"sv3d"]
    svhd = _child(buf, sv3d, b"svhd")
    proj = _child(buf, sv3d, b"proj")
    prhd = _child(buf, proj, b"prhd")
    equi = _child(buf, proj, b"equi")
    source = buf[svhd.payload + 4 : svhd.end].split(b"\x00")[0].decode("utf-8", "replace")
    yaw, pitch, roll = struct.unpack(">iii", buf[prhd.payload + 4 : prhd.payload + 16])
    bounds = struct.unpack(">4I", buf[equi.payload + 4 : equi.payload + 20])
    stereo = MONOSCOPIC
    if b"st3d" in children:
        stereo = buf[children[b"st3d"].payload + 4]
    top, bottom, left, right = (bound / 4294967296.0 for bound in bounds)
    return Spherical(
        stereo_mode=stereo,
        yaw_deg=yaw / 65536.0,
        pitch_deg=pitch / 65536.0,
        roll_deg=roll / 65536.0,
        metadata_source=source,
        bounds=(top, bottom, left, right),
    )


def dump(path: pathlib.Path) -> str:
    """The box tree around the sample entry, for looking at a file by hand.

    Offsets are the real ones in the file. Only `moov` is expanded -- `mdat` is a couple
    of gigabytes of pictures, and it is not read.
    """
    buf, moov_in_file, size = _read_moov(path)
    lines: list[str] = []

    def emit(box: _Box, depth: int, origin: int) -> None:
        name = box.kind.decode("ascii", "replace")
        lines.append(f"{'  ' * depth}{name} {box.end - box.start} bytes @ {origin + box.start}")
        if box.kind in _CONTAINERS:
            children = _walk(buf, box.payload, box.end)
        elif box.kind in _SAMPLE_ENTRIES:
            children = _walk(buf, box.payload + _VISUAL_SAMPLE_ENTRY_PAYLOAD, box.end)
        elif box.kind == b"stsd":
            children = _walk(buf, box.payload + 8, box.end)
        else:
            return
        for child in children:
            emit(child, depth + 1, origin)

    with path.open("rb") as handle:
        for box in _top_level_by_seeking(handle, size):
            if box.kind == b"moov":
                emit(_rebased(moov_in_file), 0, moov_in_file.start)
            else:
                name = box.kind.decode("ascii", "replace")
                lines.append(f"{name} {box.end - box.start} bytes @ {box.start}")
    return "\n".join(lines)
