"""The delivery matrix, encoded for real and read back: every field the spec pins down.

`{8k, 4k} x {h264, h265} x {high, mid, low} x {30, 60} fps` = 24 configurations. Each one
is encoded at its **actual delivery size** -- that is the whole point, because the level
is computed from the picture size, the sample rate and the bitrate, and a level that is
wrong (or auto-selected) is invisible in the picture and fatal to a decoder:

* `libx265` left to itself emits **Level 7.1** at 150-200 Mbps. HEVC defines nothing
  above 6.2, so no decoder will accept the file.
* `libx264 -preset ultrafast` emits **Constrained Baseline** even with `-profile:v high`.
* `hevc_nvenc` auto-selects Main tier L6.1, whose 120 Mbps ceiling 200 Mbps exceeds.

Every one of those encodes without complaint and plays back locally. Only a probe of the
written file catches them, which is why this file exists rather than a table of expected
levels next to the code that computes them.

**What is checked where.** The clips here are a handful of frames, so they contain a
single I-frame and say nothing about the GOP. The GOP structure comes from the same
`keyint` / `min-keyint` / `scenecut` parameters at every size, so it is verified in
:func:`test_the_gop_is_two_seconds_at_either_frame_rate` on small frames, where a clip
longer than a GOP is cheap. Splitting it this way keeps the matrix at about a minute;
the full-length version, with real frames and measured bitrates, is
`python tools/encode_probe.py matrix --fps 30` (and `--fps 60`).
"""

from __future__ import annotations

import functools
import pathlib

import numpy as np
import numpy.typing as npt
import pytest

from vr_compose import encode
from vr_compose.encode import BITRATES_KBPS, LADDER, SIZES, EncodeSpec

U8 = npt.NDArray[np.uint8]

try:
    TOOLS: encode.Tools | None = encode.find_tools()
except encode.FfmpegNotFound:
    TOOLS = None

needs_ffmpeg = pytest.mark.skipif(TOOLS is None, reason="ffmpeg not available")

MATRIX = [
    (size, codec, rung, fps)
    for size in SIZES
    for codec in ("h264", "h265")
    for rung in LADDER
    for fps in (30, 60)
]
MATRIX_IDS = [f"{size}-{codec}-{rung}-{fps}fps" for size, codec, rung, fps in MATRIX]

FRAMES_PER_CONFIG = 5
"""Enough for a P-frame to exist (so `bframes=0` is a real assertion), few enough that
24 configurations at up to 88 MiB per frame stay inside a minute."""


@functools.lru_cache(maxsize=2)
def _base(size: str) -> U8:
    """One gradient frame per delivery size, cached: 8K is 88 MiB to build."""
    width, height = SIZES[size]
    x = np.linspace(0, 200, width, dtype=np.float32)
    y = np.linspace(0, 55, height, dtype=np.float32)
    plane = (x[None, :] + y[:, None]).astype(np.uint8)
    return np.repeat(plane[:, :, None], 3, axis=2)


def _frame(size: str, index: int) -> U8:
    """Shifted a little each time, so there is motion for the encoder to describe."""
    return np.roll(_base(size), index * 8, axis=1)


@needs_ffmpeg
@pytest.mark.slow
@pytest.mark.parametrize(("size", "codec", "rung", "fps"), MATRIX, ids=MATRIX_IDS)
def test_every_delivery_configuration_is_conformant(
    tmp_path: pathlib.Path, size: str, codec: str, rung: str, fps: int
) -> None:
    assert TOOLS is not None
    spec = EncodeSpec.for_size(size, codec, rung, fps)
    assert spec.bitrate_kbps == BITRATES_KBPS[size][LADDER.index(rung)]

    path = tmp_path / f"{size}_{codec}_{rung}_{fps}.mp4"
    with encode.SegmentWriter(TOOLS, spec, path) as writer:
        for i in range(FRAMES_PER_CONFIG):
            writer.write(_frame(size, i))

    info = encode.probe(TOOLS, path)
    assert info.conformance_problems(spec) == [], f"{spec.describe()}: {info}"
    # ...and the same fields spelled out, so a failure names the one that moved rather
    # than handing over a list to read.
    assert info.codec == ("h264" if codec == "h264" else "hevc")
    assert info.profile == ("High" if codec == "h264" else "Main")
    assert info.tag == ("avc1" if codec == "h264" else "hvc1")
    assert (info.width, info.height) == SIZES[size]
    assert info.pix_fmt == "yuv420p"
    assert info.frames == FRAMES_PER_CONFIG
    assert info.b_frames == 0
    assert info.audio_streams == 0
    assert info.color_space == "bt709"
    assert info.color_range == "tv", "the spec is limited range; full range is opt-in"
    assert info.level == spec.level
    # A level outside the standard's own table is the failure mode this whole file exists
    # for, so it is asserted against the table rather than against `spec.level` alone.
    table = encode.H264_LEVELS if codec == "h264" else encode.H265_LEVELS
    assert info.level in [row[0] for row in table]


@needs_ffmpeg
@pytest.mark.parametrize("codec", ["h264", "h265"])
@pytest.mark.parametrize("fps", [30, 60])
def test_the_gop_is_two_seconds_at_either_frame_rate(
    tmp_path: pathlib.Path, codec: str, fps: int
) -> None:
    """Closed GOP of exactly `2 x fps`, every interval identical.

    x265 has open-GOP *on* by default and `-bf 0 -g N` alone does not close it, which
    would break the `-c copy` concat the resumable pipeline depends on. Frame sizes here
    are small because the GOP is set by the same parameters at every delivery size.
    """
    assert TOOLS is not None
    spec = EncodeSpec(64, 32, codec, 400, fps)
    assert spec.gop == 2 * fps
    rng = np.random.default_rng(3)
    path = tmp_path / f"gop_{codec}_{fps}.mp4"
    with encode.SegmentWriter(TOOLS, spec, path) as writer:
        for _ in range(2 * spec.gop + 1):
            writer.write(rng.integers(0, 256, (32, 64, 3), dtype=np.uint8))
    info = encode.probe(TOOLS, path)
    assert info.frames == 2 * spec.gop + 1
    assert info.i_intervals == (spec.gop,), "one interval, and it is the two-second GOP"
    assert info.b_frames == 0
    assert info.conformance_problems(spec) == []


@needs_ffmpeg
@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_a_scene_cut_does_not_move_a_key_frame(tmp_path: pathlib.Path, codec: str) -> None:
    """`scenecut=0` earns its place here: a cut mid-segment must not insert an I-frame.

    If it did, the segment boundaries would stop coinciding with I-frames and the
    `-c copy` join would no longer be lossless -- which is the property the whole
    resumable design rests on.
    """
    assert TOOLS is not None
    spec = EncodeSpec(64, 32, codec, 400, 30)
    path = tmp_path / f"cut_{codec}.mp4"
    with encode.SegmentWriter(TOOLS, spec, path) as writer:
        for i in range(spec.gop + 20):
            # Frame 30 is a hard cut: black before, white noise after.
            if i < 30:
                writer.write(np.zeros((32, 64, 3), np.uint8))
            else:
                writer.write(np.full((32, 64, 3), 240, np.uint8))
    info = encode.probe(TOOLS, path)
    assert info.i_intervals == (spec.gop,), "the cut must not have forced a key frame"


@pytest.mark.parametrize(("size", "codec", "rung", "fps"), MATRIX, ids=MATRIX_IDS)
def test_no_configuration_ever_leaves_the_level_to_the_encoder(
    size: str, codec: str, rung: str, fps: int
) -> None:
    """The two silent failures the delivery spec names, as assertions over all 24.

    `libx265` with an auto level picks 7.1 at the top of the ladder -- a level HEVC does
    not define -- and `libx264 -preset ultrafast` quietly drops to Constrained Baseline.
    Neither shows up in the picture, so neither may depend on the caller remembering.
    """
    spec = EncodeSpec.for_size(size, codec, rung, fps)
    args = " ".join(spec.video_args())
    if codec == "h265":
        assert f"level-idc={spec.level}" in args
        assert f"high-tier={1 if spec.tier == 'high' else 0}" in args
    else:
        assert f"-level:v {spec.level}" in args
    assert spec.preset != "ultrafast"
    with pytest.raises(ValueError, match="ultrafast"):
        EncodeSpec.for_size(size, codec, rung, fps, preset="ultrafast")


@pytest.mark.parametrize(("size", "codec", "rung", "fps"), MATRIX, ids=MATRIX_IDS)
def test_the_whole_matrix_has_a_level_inside_the_standard(
    size: str, codec: str, rung: str, fps: int
) -> None:
    """The level table, without an encoder: cheap, and it runs on machines without ffmpeg.

    HEVC stops at 6.2 and H.264 at 6.2; anything above is a number no decoder knows.
    Main tier is preferred over High tier, and 4K is kept inside the 5.x family, because
    both are what consumer decoders advertise support for.
    """
    spec = EncodeSpec.for_size(size, codec, rung, fps)
    table = encode.H264_LEVELS if codec == "h264" else encode.H265_LEVELS
    assert spec.level in [row[0] for row in table]
    assert float(spec.level) <= 6.2
    if codec == "h265":
        assert spec.tier in ("main", "high")
        if size == "4k":
            assert spec.level.startswith("5."), "a 6.x tag on a 4K file excludes 4K decoders"
            assert spec.tier == "main"
    else:
        assert spec.tier == "-"
