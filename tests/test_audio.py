"""The optional audio track (ROADMAP P14): aligned, cut, padded and faded as the user set.

The arithmetic is tested on :class:`AudioPlan` directly; everything about where a sample
actually lands is tested by joining real segments with real audio and decoding the
result, because the alignment promise is about the file a player opens, not about the
filter string that produced it.
"""

from __future__ import annotations

import hashlib
import itertools
import pathlib
import subprocess
import wave

import numpy as np
import numpy.typing as npt
import pytest

from vr_compose import audio, encode, spherical
from vr_compose.audio import AudioError, AudioPlan, AudioSpec

F32 = npt.NDArray[np.float32]

try:
    TOOLS: encode.Tools | None = encode.find_tools()
except encode.FfmpegNotFound:
    TOOLS = None

needs_ffmpeg = pytest.mark.skipif(TOOLS is None, reason="ffmpeg not available")

FPS = 30
FRAMES = 90
"""Three seconds of video: two segments, 60 + 30 frames, as the pipeline would cut them."""
RATE = 48_000
VIDEO_SAMPLES = FRAMES * RATE // FPS
SPEC = encode.EncodeSpec(64, 32, "h264", 400, FPS)
CLICK_AT = 4_800
"""0.1 s in, so the click is clear of the file's first AAC frame."""


def _plan(source_samples: int, *, channels: int = 2, source_channels: int = 2) -> AudioPlan:
    return AudioPlan(
        spec=AudioSpec(pathlib.Path("a.wav"), channels=channels),
        source_channels=source_channels,
        source_samples=source_samples,
        video_samples=VIDEO_SAMPLES,
        frame_samples=RATE // FPS,
    )


# --- the format -----------------------------------------------------------------------


def test_the_default_format_is_aac_lc_48k_stereo_192() -> None:
    spec = AudioSpec(pathlib.Path("a.wav"))
    assert spec.describe() == "AAC-LC 48 kHz stereo 192 kbps"
    assert spec.codec_args() == ["-c:a", "aac", "-profile:a", "aac_low", "-b:a", "192k"]


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"sample_rate": 22_050}, "sample rate"),
        ({"channels": 6}, "channels"),
        ({"bitrate_kbps": 16}, "bitrate"),
        ({"bitrate_kbps": 600}, "32-576 kbps"),
        ({"bitrate_kbps": 300, "sample_rate": 44_100, "channels": 1}, "32-264 kbps"),
    ],
)
def test_formats_aac_cannot_carry_are_refused(kwargs: dict[str, int], fragment: str) -> None:
    with pytest.raises(ValueError, match=fragment):
        AudioSpec(pathlib.Path("a.wav"), **kwargs)


@pytest.mark.parametrize("rate", audio.AUDIO_RATES)
@pytest.mark.parametrize("fps", [30, 60])
def test_every_rate_is_whole_samples_per_frame(rate: int, fps: int) -> None:
    """Otherwise the cut could not land exactly where the last frame ends."""
    assert rate % fps == 0


# --- the plan: cut, pad, fade ---------------------------------------------------------


def test_longer_audio_is_cut_at_the_video_and_not_faded() -> None:
    plan = _plan(5 * RATE)
    assert plan.trimmed_samples == 2 * RATE and plan.silent_samples == 0
    assert not plan.fades
    graph = plan.filter_graph()
    assert f"atrim=end_sample={VIDEO_SAMPLES}" in graph and "afade" not in graph
    assert "the last 2.00 s cut" in plan.describe()


def test_shorter_audio_fades_its_own_last_second() -> None:
    plan = _plan(2 * RATE)
    assert plan.fades and plan.fade_samples == RATE
    assert f"afade=t=out:ss={RATE}:ns={RATE}" in plan.filter_graph()
    assert f"apad=whole_len={VIDEO_SAMPLES}" in plan.filter_graph()
    assert "silent for the last 1.00 s, the audio's last 1.00 s faded out" in plan.describe()


def test_audio_under_a_second_fades_as_a_whole() -> None:
    plan = _plan(RATE // 2)
    assert plan.fade_samples == RATE // 2
    assert f"afade=t=out:ss=0:ns={RATE // 2}" in plan.filter_graph()


def test_short_by_less_than_a_frame_is_the_same_length() -> None:
    """An export "for this video" a few samples short must not lose its last second."""
    frame = RATE // FPS
    assert not _plan(VIDEO_SAMPLES - frame + 1).fades
    assert _plan(VIDEO_SAMPLES - frame).fades
    assert "same length" in _plan(VIDEO_SAMPLES - 7).describe()


@pytest.mark.parametrize(
    ("channels", "source_channels", "stage"),
    [
        (2, 1, "pan=stereo|c0=c0|c1=c0,"),
        (1, 2, "pan=mono|c0=0.5*c0+0.5*c1,"),
        (2, 2, ""),
        (2, 6, ""),
    ],
)
def test_channel_conversion(channels: int, source_channels: int, stage: str) -> None:
    graph = _plan(VIDEO_SAMPLES, channels=channels, source_channels=source_channels).filter_graph()
    layout = "stereo" if channels == 2 else "mono"
    assert f"asetpts=PTS-STARTPTS,{stage}aformat=sample_rates={RATE}:channel_layouts={layout}," in (
        graph
    )
    if not stage:
        assert "pan=" not in graph


# --- real files -----------------------------------------------------------------------


def _write_wav(path: pathlib.Path, samples: F32, rate: int = RATE, channels: int = 1) -> None:
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


def _click(path: pathlib.Path, seconds: float = 5.0) -> pathlib.Path:
    samples = np.zeros(int(seconds * RATE), np.float32)
    samples[CLICK_AT] = 0.9
    _write_wav(path, samples)
    return path


def _tone(path: pathlib.Path, samples: int, amplitude: float = 0.5) -> pathlib.Path:
    t = np.arange(samples, dtype=np.float32) / RATE
    _write_wav(path, (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32))
    return path


def _ffmpeg(*args: str) -> None:
    assert TOOLS is not None
    subprocess.run([str(TOOLS.ffmpeg), "-v", "error", "-y", *args], check=True)


@pytest.fixture(scope="module")
def segments(tmp_path_factory: pytest.TempPathFactory) -> list[pathlib.Path]:
    if TOOLS is None:
        pytest.skip("ffmpeg not available")
    directory = tmp_path_factory.mktemp("segments")
    paths = []
    for index, count in enumerate((60, 30)):
        path = directory / f"seg_{index}.mp4"
        with encode.SegmentWriter(TOOLS, SPEC, path) as writer:
            for i in range(count):
                writer.write(np.full((32, 64, 3), (index * 60 + i) * 2, np.uint8))
        paths.append(path)
    return paths


def _join(
    segments: list[pathlib.Path], source: pathlib.Path, out: pathlib.Path, **kwargs: int
) -> tuple[pathlib.Path, AudioPlan]:
    assert TOOLS is not None
    plan = audio.plan(TOOLS, AudioSpec(source, **kwargs), FRAMES, FPS)
    return encode.concat(TOOLS, segments, out, plan), plan


def _decode(path: pathlib.Path, channels: int = 2) -> F32:
    """Every decoded sample, `(n, channels)`. Includes the encoder's last-frame padding."""
    assert TOOLS is not None
    raw = subprocess.run(
        [str(TOOLS.ffmpeg), "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "f32le", "-"],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, "<f4").reshape(-1, channels)


@needs_ffmpeg
def test_the_plan_counts_the_samples_the_mux_will_see(tmp_path: pathlib.Path) -> None:
    """2.5 s at 44.1 kHz mono is 120 000 samples once resampled to 48 kHz."""
    assert TOOLS is not None
    source = tmp_path / "a.wav"
    _write_wav(source, np.zeros(110_250, np.float32), rate=44_100)
    plan = audio.plan(TOOLS, AudioSpec(source), FRAMES, FPS)
    assert (plan.source_samples, plan.source_channels) == (120_000, 1)
    assert (plan.video_samples, plan.frame_samples) == (VIDEO_SAMPLES, 1600)


@needs_ffmpeg
def test_a_file_that_cannot_be_used_is_refused_before_anything_runs(
    tmp_path: pathlib.Path, segments: list[pathlib.Path]
) -> None:
    assert TOOLS is not None
    with pytest.raises(AudioError, match="no such audio file"):
        audio.plan(TOOLS, AudioSpec(tmp_path / "missing.wav"), FRAMES, FPS)
    with pytest.raises(AudioError, match="no audio track"):
        audio.plan(TOOLS, AudioSpec(segments[0]), FRAMES, FPS)  # video only
    text = tmp_path / "notes.wav"
    text.write_text("not audio", encoding="utf-8")
    with pytest.raises(AudioError, match="cannot read"):
        audio.plan(TOOLS, AudioSpec(text), FRAMES, FPS)


@needs_ffmpeg
@pytest.mark.parametrize("kind", ["wav", "mp3", "late-flac"])
def test_the_first_sample_lands_on_the_first_frame(
    tmp_path: pathlib.Path, segments: list[pathlib.Path], kind: str
) -> None:
    """Plain, with an encoder delay to take out (mp3), and starting 0.5 s into its own
    timeline (`late-flac`): each puts the click 0.1 s after the first frame."""
    assert TOOLS is not None
    source = _click(tmp_path / "c.wav")
    if kind == "mp3":
        if "libmp3lame" not in TOOLS.encoders:
            pytest.skip("this ffmpeg has no MP3 encoder to make the fixture with")
        encoded = tmp_path / "c.mp3"
        _ffmpeg("-i", str(source), "-c:a", "libmp3lame", str(encoded))
        source = encoded
    elif kind == "late-flac":
        encoded = tmp_path / "late.mka"
        _ffmpeg("-itsoffset", "0.5", "-i", str(source), "-c:a", "flac", str(encoded))
        source = encoded
    out, _plan = _join(segments, source, tmp_path / "out.mp4")
    peak = int(np.argmax(np.abs(_decode(out)[:, 0])))
    assert abs(peak - CLICK_AT) <= RATE // 1000, f"click at sample {peak}, want {CLICK_AT}"


@needs_ffmpeg
def test_longer_audio_ends_with_the_video_and_conforms(
    tmp_path: pathlib.Path, segments: list[pathlib.Path]
) -> None:
    assert TOOLS is not None
    out, plan = _join(segments, _tone(tmp_path / "long.wav", 5 * RATE), tmp_path / "out.mp4")
    assert plan.trimmed_samples == 2 * RATE
    info = encode.probe(TOOLS, out)
    assert info.frames == FRAMES and info.audio_streams == 1
    assert abs(info.audio_duration - FRAMES / FPS) <= audio.AAC_FRAME / RATE
    assert info.conformance_problems(SPEC, audio=plan.spec) == [], info
    # ...and a check that does not expect audio says so, rather than passing it quietly.
    assert any("audio" in p for p in info.conformance_problems(SPEC))


@needs_ffmpeg
def test_shorter_audio_fades_over_its_last_second_then_goes_silent(
    tmp_path: pathlib.Path, segments: list[pathlib.Path]
) -> None:
    out, plan = _join(segments, _tone(tmp_path / "short.wav", 2 * RATE), tmp_path / "out.mp4")
    assert plan.fades
    left = _decode(out)[:, 0]
    window = RATE // 10
    peaks = [float(np.abs(left[i : i + window]).max()) for i in range(0, 2 * RATE, window)]
    full = peaks[0]
    assert all(abs(p - full) < 0.02 for p in peaks[:9]), peaks  # untouched before the fade
    assert all(b < a for a, b in itertools.pairwise(peaks[10:])), peaks  # falling
    assert peaks[-1] < 0.1 * full, peaks
    assert float(np.abs(left[2 * RATE + audio.AAC_FRAME : VIDEO_SAMPLES]).max()) < 1e-3


@needs_ffmpeg
def test_audio_a_few_samples_short_keeps_its_level_to_the_end(
    tmp_path: pathlib.Path, segments: list[pathlib.Path]
) -> None:
    source = _tone(tmp_path / "almost.wav", VIDEO_SAMPLES - 100)
    out, plan = _join(segments, source, tmp_path / "out.mp4")
    assert not plan.fades
    left = _decode(out)[:, 0]
    head = float(np.abs(left[RATE // 10 : RATE // 5]).max())
    tail = float(np.abs(left[VIDEO_SAMPLES - RATE // 10 - 100 : VIDEO_SAMPLES - 200]).max())
    assert abs(tail - head) < 0.02, (head, tail)


@needs_ffmpeg
def test_mono_is_copied_to_both_channels_at_full_level(
    tmp_path: pathlib.Path, segments: list[pathlib.Path]
) -> None:
    """ffmpeg's own mono-to-stereo upmix is 3 dB down; ours is a copy."""
    out, _plan = _join(segments, _tone(tmp_path / "m.wav", 5 * RATE), tmp_path / "out.mp4")
    stereo = _decode(out)[RATE // 10 : RATE]
    assert np.allclose(stereo[:, 0], stereo[:, 1], atol=1e-4)
    assert abs(float(np.abs(stereo[:, 0]).max()) - 0.5) < 0.02


@needs_ffmpeg
def test_a_non_default_format_is_written_as_asked(
    tmp_path: pathlib.Path, segments: list[pathlib.Path]
) -> None:
    assert TOOLS is not None
    out, plan = _join(
        segments,
        _tone(tmp_path / "s.wav", 5 * RATE),
        tmp_path / "out.mp4",
        sample_rate=44_100,
        channels=1,
        bitrate_kbps=128,
    )
    info = encode.probe(TOOLS, out)
    assert (info.audio_sample_rate, info.audio_channels) == (44_100, 1)
    assert info.conformance_problems(SPEC, audio=plan.spec) == [], info


@needs_ffmpeg
def test_the_same_audio_joins_to_the_same_bytes(
    tmp_path: pathlib.Path, segments: list[pathlib.Path]
) -> None:
    """A resumed run re-does the join, so it has to come out identical."""
    source = _tone(tmp_path / "t.wav", 4 * RATE)
    first, _ = _join(segments, source, tmp_path / "a.mp4")
    second, _ = _join(segments, source, tmp_path / "b.mp4")
    assert (
        hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()
    )


@needs_ffmpeg
def test_without_audio_the_join_is_the_command_it_always_was(
    tmp_path: pathlib.Path, segments: list[pathlib.Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert TOOLS is not None
    seen: list[list[str]] = []
    real = encode._run_concat

    def spy(command: list[str]) -> subprocess.CompletedProcess[str]:
        seen.append(command)
        return real(command)

    monkeypatch.setattr(encode, "_run_concat", spy)
    out = encode.concat(TOOLS, segments, tmp_path / "plain.mp4")
    listing = seen[0][seen[0].index("-i") + 1]
    assert seen[0] == [
        str(TOOLS.ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
        "-safe", "0", "-i", listing, "-c", "copy", "-movflags", "+faststart",
        "-fflags", "+bitexact", "-flags", "+bitexact", "-f", "mp4", f"{out}.part",
    ]  # fmt: skip
    assert encode.probe(TOOLS, out).audio_streams == 0


def _stream_md5(path: pathlib.Path, selector: str) -> str:
    assert TOOLS is not None
    return subprocess.run(
        [str(TOOLS.ffmpeg), "-v", "error", "-i", str(path), "-map", selector, "-f", "md5", "-"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@needs_ffmpeg
def test_the_360_metadata_goes_in_beside_an_audio_track(
    tmp_path: pathlib.Path, segments: list[pathlib.Path]
) -> None:
    """Injection grows `moov` and shifts every chunk offset, the audio track's included."""
    assert TOOLS is not None
    out, plan = _join(segments, _tone(tmp_path / "t.wav", 4 * RATE), tmp_path / "out.mp4")
    before = (_stream_md5(out, "0:v:0"), _stream_md5(out, "0:a:0"))
    spherical.inject(out)
    assert (_stream_md5(out, "0:v:0"), _stream_md5(out, "0:a:0")) == before
    info = encode.probe(TOOLS, out)
    assert info.projection == "equirectangular"
    assert info.conformance_problems(SPEC, spherical=True, audio=plan.spec) == [], info
