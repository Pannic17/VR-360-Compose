"""Encoder configuration and MP4 read-back.

The level table tests pin the matrix recorded in AGENTS.md §5; the integration tests need
ffmpeg and are skipped where it is absent.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import numpy as np
import pytest

from vr_compose import encode
from vr_compose.encode import EncodeSpec, StreamInfo, select_level


@pytest.mark.parametrize(
    ("size", "codec", "kbps", "fps", "level", "tier"),
    [
        # AGENTS.md §5, verified with ffprobe on real encodes
        ("8k", "h264", 200_000, 30, "6.0", "-"),
        ("8k", "h264", 100_000, 30, "6.0", "-"),
        ("8k", "h264", 200_000, 60, "6.1", "-"),
        ("8k", "h265", 200_000, 30, "6.2", "main"),
        ("8k", "h265", 150_000, 30, "6.2", "main"),
        ("8k", "h265", 100_000, 30, "6.1", "main"),
        ("4k", "h264", 57_000, 30, "5.1", "-"),
        ("4k", "h264", 57_000, 60, "5.2", "-"),
        ("4k", "h265", 57_000, 30, "5.2", "main"),
        ("4k", "h265", 43_000, 30, "5.2", "main"),
        ("4k", "h265", 28_000, 30, "5.1", "main"),
        # the un-scaled 4k ladder forced High tier; the scaled one does not. 100 Mbps sits
        # exactly on L5.0 High's cap (100,000 kbps) and the standard's bound is inclusive.
        ("4k", "h265", 200_000, 30, "5.2", "high"),
        ("4k", "h265", 100_000, 30, "5.0", "high"),
        ("4k", "h265", 100_001, 30, "5.1", "high"),
    ],
)
def test_select_level_matches_the_recorded_matrix(
    size: str, codec: str, kbps: int, fps: int, level: str, tier: str
) -> None:
    width, height = encode.SIZES[size]
    assert select_level(width, height, codec, kbps, fps) == (level, tier)


def test_select_level_rejects_the_impossible() -> None:
    with pytest.raises(ValueError, match="no HEVC level"):
        select_level(7680, 7680, "h265", 200_000, 30)  # 59 Mpx > L6.2's 35.65 Mpx
    with pytest.raises(ValueError, match="unknown codec"):
        select_level(1920, 960, "av1", 10_000, 30)


def test_h264_macroblock_count_rounds_up() -> None:
    # 7680x3840 = 115,200 MBs, above L5.2's 36,864: 8K H.264 needs 6.0 by frame size alone.
    assert select_level(7680, 3840, "h264", 1_000, 30)[0] == "6.0"
    assert select_level(4096, 2048, "h264", 1_000, 30)[0] == "5.1"


def test_bitrate_ladders_hold_bits_per_pixel() -> None:
    """The 4k ladder is the 8k ladder scaled by the pixel ratio 3.515625."""
    ratio = (7680 * 3840) / (4096 * 2048)
    for high, low in zip(encode.BITRATES_KBPS["8k"], encode.BITRATES_KBPS["4k"], strict=True):
        assert high / low == pytest.approx(ratio, rel=0.02)


def test_for_size_picks_the_ladder_rung() -> None:
    spec = EncodeSpec.for_size("4k", "h265", "mid", 30)
    assert (spec.width, spec.height, spec.bitrate_kbps) == (4096, 2048, 43_000)
    assert spec.gop == 60
    assert EncodeSpec.for_size("8k", "h264", "high", 60).gop == 120


def test_spec_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="ultrafast"):
        EncodeSpec(7680, 3840, "h264", 200_000, 30, preset="ultrafast")
    with pytest.raises(ValueError, match="codec"):
        EncodeSpec(7680, 3840, "vp9", 200_000, 30)
    with pytest.raises(ValueError, match="2:1"):
        EncodeSpec(7680, 3000, "h264", 200_000, 30)
    with pytest.raises(ValueError, match="positive"):
        EncodeSpec(7680, 3840, "h264", 0, 30)


def test_spec_rejects_an_impossible_master() -> None:
    with pytest.raises(ValueError, match="2:1 too"):
        EncodeSpec(4096, 2048, "h264", 50_000, 30, master=(7680, 3000))
    with pytest.raises(ValueError, match="smaller than the delivery size"):
        EncodeSpec(7680, 3840, "h264", 200_000, 30, master=(4096, 2048))


def test_master_defaults_to_the_delivery_size() -> None:
    """No master means the stitcher renders the delivered size, i.e. today's 8K path."""
    spec = EncodeSpec.for_size("8k", "h264", "high", 30)
    assert (spec.master_width, spec.master_height) == (7680, 3840)
    assert not spec.downsamples


def test_ffmpeg_gets_its_own_process_group() -> None:
    """So a console Ctrl-C reaches only us. Without this the encoder dies first and
    cancellation is reported as an encoder failure -- see child_creation_flags."""
    flags = encode.child_creation_flags()
    if sys.platform == "win32":
        assert flags & subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        assert flags == 0


def _joined(args: list[str]) -> str:
    return " ".join(args)


def test_h264_args_carry_every_pinned_field() -> None:
    args = _joined(EncodeSpec.for_size("8k", "h264", "high", 30).video_args())
    for required in (
        "-c:v libx264",
        "-profile:v high",
        "-level:v 6.0",
        "bframes=0",
        "open-gop=0",
        "keyint=60",
        "min-keyint=60",
        "scenecut=0",
        "-tag:v avc1",
        "-pix_fmt yuv420p",
        "-an",
        "out_color_matrix=bt709",
        "-b:v 200000k",
        "-maxrate 200000k",
        "-bufsize 400000k",
    ):
        assert required in args, required


def test_h265_args_pin_level_and_tier() -> None:
    args = _joined(EncodeSpec.for_size("8k", "h265", "high", 30).video_args())
    assert "-c:v libx265" in args
    assert "level-idc=6.2" in args
    assert "high-tier=0" in args
    assert "-tag:v hvc1" in args
    four_k = _joined(EncodeSpec.for_size("4k", "h265", "low", 30).video_args())
    assert "level-idc=5.1" in four_k and "high-tier=0" in four_k


def test_range_flag_follows_the_spec() -> None:
    limited = _joined(EncodeSpec(4096, 2048, "h264", 50_000, 30).video_args())
    assert "-color_range tv" in limited and "out_range=limited" in limited
    full = _joined(EncodeSpec(4096, 2048, "h264", 50_000, 30, full_range=True).video_args())
    assert "-color_range pc" in full and "out_range=full" in full


def test_input_args_describe_raw_rgb_on_stdin() -> None:
    args = _joined(EncodeSpec(4096, 2048, "h264", 50_000, 60).input_args())
    assert "-f rawvideo" in args and "-pix_fmt rgb24" in args
    assert "-s 4096x2048" in args and "-r 60" in args and args.endswith("-i -")


def test_a_master_is_fed_whole_and_resampled_inside_the_encode() -> None:
    """16K in, 8K out: stdin carries the master, one swscale pass delivers the spec size.

    The resize must sit in the same `scale` filter as the range and matrix conversion --
    a second scale would resample twice, and resizing after `format=yuv420p` would
    decimate the chroma before averaging it.
    """
    spec = EncodeSpec.for_size("8k", "h264", "high", 30, master=(15360, 7680))
    assert spec.downsamples and (spec.master_width, spec.master_height) == (15360, 7680)
    assert "-s 15360x7680" in _joined(spec.input_args()), "the master goes in"
    args = _joined(spec.video_args())
    assert "scale=7680:3840:flags=lanczos:in_range=full" in args, "one pass, lanczos, then yuv"
    assert args.count("scale=") == 1, "a second scale filter would resample twice"
    assert args.index("scale=") < args.index("format=yuv420p"), "resize before chroma decimation"
    # the level is the delivery size's, not the master's -- no level admits 16K at all
    assert "-level:v 6.0" in args


def test_the_4k_delivery_comes_off_the_8k_master() -> None:
    """ROADMAP P3: warping straight to 4096x2048 aliases; resample the master instead."""
    spec = EncodeSpec.for_size("4k", "h265", "high", 30, master=(7680, 3840))
    assert "-s 7680x3840" in _joined(spec.input_args())
    assert "scale=4096:2048:flags=lanczos" in _joined(spec.video_args())
    assert "level-idc=5.2" in _joined(spec.video_args()), "the level follows the delivery size"


def test_no_resize_filter_without_a_master() -> None:
    args = _joined(EncodeSpec.for_size("8k", "h264", "high", 30).video_args())
    assert "flags=lanczos" not in args and "scale=in_range=full" in args


def test_encoder_threads_caps_the_footprint_per_codec() -> None:
    """Measured: x264's commit follows *frame* threads, so the cap switches it to slice
    threading; x265 needs its own knobs. Reproduce with tools/encode_probe.py memory."""
    x264 = _joined(EncodeSpec.for_size("8k", "h264", "high", 30, encoder_threads=16).video_args())
    assert "sliced-threads=1:threads=16" in x264
    x265 = _joined(EncodeSpec.for_size("8k", "h265", "high", 30, encoder_threads=8).video_args())
    assert "pools=8:frame-threads=2" in x265
    # unset means untouched: no thread parameter at all
    for codec in ("h264", "h265"):
        args = _joined(EncodeSpec.for_size("8k", codec, "high", 30).video_args())
        assert "threads=" not in args and "pools=" not in args, codec


def test_encoder_threads_is_not_a_determinism_knob() -> None:
    """Slice threading measured non-reproducible, so the two flags must stay independent:
    asking for a smaller footprint must not silently drop the determinism pin."""
    spec = EncodeSpec.for_size("8k", "h264", "high", 30, encoder_threads=16, deterministic=True)
    args = _joined(spec.video_args())
    assert "threads=1" in args, "deterministic still pins threads=1"
    with pytest.raises(ValueError, match="encoder_threads"):
        EncodeSpec.for_size("8k", "h264", "high", 30, encoder_threads=0)


def test_describe_is_human_readable() -> None:
    text = EncodeSpec.for_size("4k", "h265", "high", 30).describe()
    assert "h265 4096x2048" in text and "57 Mbps" in text and "main tier" in text


def _info(**overrides: object) -> StreamInfo:
    base = dict(
        codec="h264", profile="High", level="6.0", tag="avc1", width=7680, height=3840,
        pix_fmt="yuv420p", color_range="tv", color_space="bt709", frames=120, b_frames=0,
        i_intervals=(60,), bitrate_mbps=198.0, audio_streams=0,
    )  # fmt: skip
    base.update(overrides)
    return StreamInfo(**base)  # type: ignore[arg-type]


def test_conformance_passes_a_correct_stream() -> None:
    spec = EncodeSpec.for_size("8k", "h264", "high", 30)
    assert _info().conformance_problems(spec) == []


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"profile": "Constrained Baseline"}, "profile"),
        ({"level": "7.1"}, "level"),
        ({"tag": "hev1"}, "tag"),
        ({"pix_fmt": "yuv444p"}, "pix_fmt"),
        ({"b_frames": 3}, "B-frames"),
        ({"i_intervals": (60, 61)}, "I-frame intervals"),
        ({"audio_streams": 1}, "audio"),
        ({"width": 4096, "height": 2048}, "size"),
    ],
)
def test_conformance_names_each_violation(override: dict[str, object], fragment: str) -> None:
    spec = EncodeSpec.for_size("8k", "h264", "high", 30)
    problems = _info(**override).conformance_problems(spec)
    assert len(problems) == 1 and fragment in problems[0], problems


def test_deterministic_flag_pins_each_encoder_to_its_reproducible_mode() -> None:
    """Measured: x265 needs frame-threads=1:wpp=0; x264 needs threads=1 for short segments."""
    plain_265 = _joined(EncodeSpec(4096, 2048, "h265", 57_000, 30).video_args())
    strict_265 = _joined(
        EncodeSpec(4096, 2048, "h265", 57_000, 30, deterministic=True).video_args()
    )
    assert "wpp=0" not in plain_265 and "threads" not in plain_265
    assert "frame-threads=1:wpp=0" in strict_265

    plain_264 = _joined(EncodeSpec(4096, 2048, "h264", 57_000, 30).video_args())
    strict_264 = _joined(
        EncodeSpec(4096, 2048, "h264", 57_000, 30, deterministic=True).video_args()
    )
    assert "threads=" not in plain_264, "auto threads: fast, and reproducible for long segments"
    assert ":threads=1" in strict_264 and "wpp" not in strict_264


# --- integration: needs ffmpeg -------------------------------------------------------

try:
    TOOLS: encode.Tools | None = encode.find_tools()
except encode.FfmpegNotFound:
    TOOLS = None

needs_ffmpeg = pytest.mark.skipif(TOOLS is None, reason="ffmpeg not available")


@needs_ffmpeg
def test_find_tools_reports_the_required_encoders() -> None:
    assert TOOLS is not None
    assert TOOLS.version.startswith("ffmpeg version")
    TOOLS.require("libx264", "libx265")
    with pytest.raises(encode.FfmpegNotFound, match="lacks"):
        TOOLS.require("definitely_not_an_encoder")


@needs_ffmpeg
@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_segment_probe_and_concat_round_trip(tmp_path: pathlib.Path, codec: str) -> None:
    """Write two tiny segments, read them back, join them, read the join back."""
    assert TOOLS is not None
    spec = EncodeSpec(64, 32, codec, 400, 30)
    rng = np.random.default_rng(1)

    def frame(i: int) -> np.ndarray:
        # smooth gradients plus a little noise: enough for the encoder to work on
        base = np.linspace(0, 255, 64, dtype=np.float32)[None, :, None]
        return np.clip(base + i * 3 + rng.normal(0, 4, (32, 64, 3)), 0, 255).astype(np.uint8)

    segments = []
    for seg in range(2):
        path = tmp_path / f"seg_{seg}.mp4"
        with encode.SegmentWriter(TOOLS, spec, path) as writer:
            for i in range(60):
                writer.write(frame(seg * 60 + i))
        assert path.exists() and not writer.partial.exists()
        info = encode.probe(TOOLS, path)
        assert info.frames == 60
        assert info.conformance_problems(spec) == [], info
        segments.append(path)

    joined = encode.concat(TOOLS, segments, tmp_path / "out.mp4")
    info = encode.probe(TOOLS, joined)
    assert info.frames == 120
    assert info.i_intervals == (60,), "segment boundaries must land exactly on GOP boundaries"
    assert info.conformance_problems(spec) == [], info
    assert info.color_space == "bt709"


@needs_ffmpeg
def test_segment_writer_rejects_wrong_frames_and_cleans_up(tmp_path: pathlib.Path) -> None:
    assert TOOLS is not None
    spec = EncodeSpec(64, 32, "h264", 400, 30)
    writer = encode.SegmentWriter(TOOLS, spec, tmp_path / "bad.mp4")
    with pytest.raises(ValueError, match="frame must be"):
        writer.write(np.zeros((32, 64, 4), np.uint8))
    writer.abort()
    assert not writer.partial.exists() and not writer.path.exists()


@needs_ffmpeg
def test_a_downsampling_writer_wants_masters_not_delivery_frames(tmp_path: pathlib.Path) -> None:
    assert TOOLS is not None
    spec = EncodeSpec(64, 32, "h264", 400, 30, master=(128, 64))
    writer = encode.SegmentWriter(TOOLS, spec, tmp_path / "bad.mp4")
    with pytest.raises(ValueError, match=r"frame must be \(64, 128, 3\)"):
        writer.write(np.zeros((32, 64, 3), np.uint8))  # the delivery size, not the master's
    writer.abort()
