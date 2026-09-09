"""The `st3d` / `sv3d` writer: box layout, and the fix-ups a wrong one would silently skip.

Two of these matter more than the rest. `test_the_pixels_are_untouched...` is the guard on
the chunk-offset patch: get that wrong and the file still parses, still reports its
metadata and still shows a duration, but decodes to nothing. And the ffprobe assertions
are there because agreeing with our own parser proves nothing -- ffmpeg's `mov` demuxer is
a separate implementation of the same spec.
"""

from __future__ import annotations

import pathlib
import struct
import subprocess

import numpy as np
import pytest

from vr_compose import encode, spherical
from vr_compose.encode import EncodeSpec

try:
    TOOLS: encode.Tools | None = encode.find_tools()
except encode.FfmpegNotFound:
    TOOLS = None

needs_ffmpeg = pytest.mark.skipif(TOOLS is None, reason="ffmpeg not available")


def _boxes(payload: bytes) -> dict[bytes, bytes]:
    """Split a concatenation of boxes into {type: payload}, one level deep."""
    out: dict[bytes, bytes] = {}
    pos = 0
    while pos < len(payload):
        size = int.from_bytes(payload[pos : pos + 4], "big")
        out[payload[pos + 4 : pos + 8]] = payload[pos + 8 : pos + size]
        pos += size
    return out


def _write_mp4(path: pathlib.Path, *, codec: str = "h264", faststart: bool = True) -> pathlib.Path:
    """A small but genuine delivery file, written by the pipeline's own encoder path."""
    assert TOOLS is not None
    spec = EncodeSpec(64, 32, codec, 400, 30)
    rng = np.random.default_rng(7)
    with encode.SegmentWriter(TOOLS, spec, path) as writer:
        for i in range(6):
            base = np.linspace(0, 255, 64, dtype=np.float32)[None, :, None]
            noisy = base + i * 9 + rng.normal(0, 3, (32, 64, 3))
            writer.write(np.clip(noisy, 0, 255).astype(np.uint8))
    if not faststart:
        # The writer always asks for +faststart, i.e. moov first. Re-muxing without it
        # puts mdat first, which is the layout where *no* chunk offset needs patching.
        plain = path.with_name("plain_" + path.name)
        subprocess.run(
            [str(TOOLS.ffmpeg), "-hide_banner", "-v", "error", "-y", "-i", str(path),
             "-c", "copy", "-movflags", "-faststart", str(plain)],
            check=True,
        )  # fmt: skip
        return plain
    return path


def _decode_md5(path: pathlib.Path) -> str:
    assert TOOLS is not None
    result = subprocess.run(
        [str(TOOLS.ffmpeg), "-hide_banner", "-v", "error", "-i", str(path), "-f", "md5", "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


# --- the boxes themselves: no ffmpeg needed ------------------------------------------


def test_the_box_tree_matches_the_spherical_video_v2_layout() -> None:
    payload = spherical.Spherical(yaw_deg=90.0, pitch_deg=-45.5, roll_deg=1.0).boxes()
    top = _boxes(payload)
    assert list(top) == [b"st3d", b"sv3d"]
    assert top[b"st3d"] == b"\x00\x00\x00\x00" + bytes([spherical.MONOSCOPIC])
    sv3d = _boxes(top[b"sv3d"])
    assert list(sv3d) == [b"svhd", b"proj"]
    assert sv3d[b"svhd"] == b"\x00\x00\x00\x00" + b"VR-Compose\x00"
    proj = _boxes(sv3d[b"proj"])
    assert list(proj) == [b"prhd", b"equi"]
    # 16.16 fixed point, signed: a negative pitch has to survive as a negative number.
    assert struct.unpack(">iii", proj[b"prhd"][4:]) == (90 * 65536, -45.5 * 65536, 65536)
    assert struct.unpack(">4I", proj[b"equi"][4:]) == (0, 0, 0, 0)


def test_every_box_declares_its_own_length() -> None:
    payload = spherical.Spherical().boxes()
    pos = 0
    seen = 0
    while pos < len(payload):
        size = int.from_bytes(payload[pos : pos + 4], "big")
        assert size >= 8, "a box cannot be shorter than its header"
        pos += size
        seen += 1
    assert pos == len(payload), "the boxes must tile the payload exactly"
    assert seen == 2


def test_bounds_become_0_32_fixed_point() -> None:
    payload = spherical.Spherical(bounds=(0.25, 0.0, 0.5, 0.0)).boxes()
    equi = _boxes(_boxes(_boxes(payload)[b"sv3d"])[b"proj"])[b"equi"]
    assert struct.unpack(">4I", equi[4:]) == (1 << 30, 0, 1 << 31, 0)


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"stereo_mode": 3}, "stereo_mode"),
        ({"bounds": (0.0, 0.0, 0.0, 1.0)}, "in \\[0, 1\\)"),
        ({"bounds": (0.6, 0.5, 0.0, 0.0)}, "no picture"),
    ],
)
def test_impossible_metadata_is_refused(kwargs: dict[str, object], fragment: str) -> None:
    with pytest.raises(ValueError, match=fragment):
        spherical.Spherical(**kwargs)  # type: ignore[arg-type]


def test_a_file_that_is_not_an_mp4_is_refused(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "nope.mp4"
    path.write_bytes(b"\x00" * 64)
    with pytest.raises(spherical.SphericalError, match="is this an MP4"):
        spherical.inject(path)


# --- against real files --------------------------------------------------------------


@needs_ffmpeg
@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_the_metadata_round_trips_through_a_real_file(tmp_path: pathlib.Path, codec: str) -> None:
    path = _write_mp4(tmp_path / f"{codec}.mp4", codec=codec)
    assert spherical.read(path) is None
    wanted = spherical.Spherical(
        stereo_mode=spherical.STEREO_TOP_BOTTOM,
        yaw_deg=12.5,
        pitch_deg=-3.25,
        roll_deg=0.5,
        metadata_source="test",
        bounds=(0.125, 0.25, 0.0, 0.5),
    )
    added = spherical.inject(path, wanted)
    assert added > 0
    got = spherical.read(path)
    assert got is not None
    assert (got.stereo_mode, got.metadata_source) == (wanted.stereo_mode, "test")
    assert (got.yaw_deg, got.pitch_deg, got.roll_deg) == (12.5, -3.25, 0.5)
    assert got.bounds == pytest.approx(wanted.bounds)
    # The boxes land inside the *sample entry*, which is where a player looks for them.
    tree = spherical.dump(path)
    assert "sv3d" in tree and "st3d" in tree
    entry = "avc1" if codec == "h264" else "hvc1"
    assert tree.index(entry) < tree.index("sv3d") < tree.index("stts")


@needs_ffmpeg
@pytest.mark.parametrize("faststart", [True, False])
def test_the_pixels_are_untouched_whichever_way_the_file_is_laid_out(
    tmp_path: pathlib.Path, faststart: bool
) -> None:
    """The chunk-offset guard.

    With `+faststart` the metadata is inserted *before* mdat, so every chunk offset moves;
    without it, mdat comes first and none do. Both layouts must decode to the same pixels
    they did before, and only a decode proves it -- a corrupt offset table still parses.
    """
    path = _write_mp4(tmp_path / "layout.mp4", faststart=faststart)
    before = _decode_md5(path)
    spherical.inject(path)
    assert _decode_md5(path) == before


@needs_ffmpeg
def test_injecting_twice_does_not_write_the_boxes_twice(tmp_path: pathlib.Path) -> None:
    path = _write_mp4(tmp_path / "once.mp4")
    first = spherical.inject(path)
    size = path.stat().st_size
    assert spherical.inject(path) == 0
    assert path.stat().st_size == size
    assert spherical.dump(path).count("sv3d") == 1
    assert first == 104, "the default metadata is a fixed 104 bytes; a change here is a spec change"


@needs_ffmpeg
def test_the_file_is_never_slurped_into_memory(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A delivery MP4 is about 2 GB; only `moov` (a few KB) may be held (AGENTS.md 2.7).

    Pinned by forbidding `read_bytes` outright rather than by measuring memory, which
    would be a flaky assertion about the machine rather than about this code. Measured
    on a 124 MiB file: peak commit grew 16 MiB, the copy buffer.
    """
    path = _write_mp4(tmp_path / "streamed.mp4")

    def refuse(self: pathlib.Path) -> bytes:
        raise AssertionError(f"read the whole of {self.name} into memory")

    monkeypatch.setattr(pathlib.Path, "read_bytes", refuse)
    assert spherical.inject(path) == 104
    assert spherical.read(path) is not None
    assert "sv3d" in spherical.dump(path)


@needs_ffmpeg
def test_ffprobe_reads_the_projection_back(tmp_path: pathlib.Path) -> None:
    """Cross-validation: ffmpeg's demuxer is an implementation we did not write."""
    assert TOOLS is not None
    path = _write_mp4(tmp_path / "probe.mp4")
    assert encode.probe(TOOLS, path).projection == ""
    spherical.inject(path)
    assert encode.probe(TOOLS, path).projection == "equirectangular"


@needs_ffmpeg
def test_conformance_wants_the_metadata_only_when_it_is_expected(tmp_path: pathlib.Path) -> None:
    assert TOOLS is not None
    spec = EncodeSpec(64, 32, "h264", 400, 30)
    path = _write_mp4(tmp_path / "conformance.mp4")
    bare = encode.probe(TOOLS, path)
    assert bare.conformance_problems(spec) == []
    assert bare.conformance_problems(spec, spherical=True) == ["spherical projection 'absent'"]
    spherical.inject(path)
    assert encode.probe(TOOLS, path).conformance_problems(spec, spherical=True) == []


@needs_ffmpeg
def test_a_failed_rewrite_leaves_the_original_in_place(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The temporary-then-rename contract: a delivery file is never half-written.

    The failure is injected at the write itself -- a full disk, or an interrupt -- because
    that is the only moment at which a partial file can exist.
    """
    path = _write_mp4(tmp_path / "atomic.mp4")
    original = path.read_bytes()

    def die_partway(source: object, target: object, start: int, length: int) -> None:
        target.write(b"half a block")  # type: ignore[attr-defined]
        raise OSError("no space left on device")

    monkeypatch.setattr(spherical, "_copy", die_partway)
    with pytest.raises(OSError, match="no space"):
        spherical.inject(path)
    monkeypatch.undo()
    assert path.read_bytes() == original
    assert not path.with_name(path.name + ".sv3d").exists()
    assert spherical.read(path) is None
