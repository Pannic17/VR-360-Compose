"""The delivery's optional audio track (ROADMAP P14): one file, aligned, cut to fit, AAC.

The user's rules (2026-10-14):

* **Only when asked.** No `--audio` means no audio track, and the delivery is byte for
  byte what it was before this module existed.
* **First sample on the first frame** -- the first frame *of this output*, so a run of
  `--frames 1000-` puts the audio's start on frame 1000. Whatever start offset the file
  carries (an m4a's `start_time`, an mp3's encoder delay) is taken out.
* **Longer than the video:** cut at the video's last frame, to the sample. No fade there:
  the picture ends at the same instant, and the user asked for an envelope only on the
  other side.
* **Shorter:** silence after it, and the audio's own last second faded out so it does not
  stop mid-sound. A file short by less than one video frame counts as the same length --
  audio exported "for this video" is routinely a few samples short, and fading the last
  second of it over that would damage what was meant to play.
* **Format:** AAC-LC, 48 kHz, stereo, 192 kbps unless told otherwise. Mono is copied to
  both channels at full level (ffmpeg's own upmix is 3 dB down); stereo to mono averages.

**Where it happens.** Not in the segments: they stay video-only, so resuming is untouched
and a different audio file on a resumed run only changes the join. The track is added by
the concat that joins the segments (:func:`vr_compose.encode.concat`), with the video
copied, which saves writing an 8K delivery twice.

**Why the source is decoded once up front.** The fade has to know where the audio ends,
and container durations are estimates for half the formats people use (ADTS AAC is a
guess from the bitrate). Decoding to 8-bit mono and counting bytes is exact and costs a
second or two per hour of audio. The count runs the same alignment filters as the mux
(:func:`_align`), so the two cannot disagree about where sample N is.

**Measured** (2026-10-14, a click at 0.1 s): WAV, MP3, FLAC at 44.1 kHz, Ogg Vorbis, AAC
in m4a, and PCM / FLAC with a 0.5 s start offset all put the click on output sample 4800
exactly. One file shape does not: AAC in an MP4 whose edit list starts at the encoder's
priming rather than after it -- what ffmpeg's own `-itsoffset` writes -- has those 1024
samples as its first ones, so its content lands 21 ms (under one frame) late.
"""

from __future__ import annotations

import dataclasses
import fractions
import json
import pathlib
import subprocess
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vr_compose.encode import Tools

__all__ = [
    "AUDIO_CHANNELS",
    "AUDIO_EXTENSIONS",
    "AUDIO_RATES",
    "DEFAULT_AUDIO_BITRATE_KBPS",
    "DEFAULT_AUDIO_CHANNELS",
    "DEFAULT_AUDIO_RATE",
    "AudioError",
    "AudioPlan",
    "AudioSpec",
    "plan",
]

DEFAULT_AUDIO_BITRATE_KBPS = 192
DEFAULT_AUDIO_RATE = 48_000
DEFAULT_AUDIO_CHANNELS = 2

AUDIO_RATES = (44_100, 48_000, 96_000)
"""Each is a whole number of samples per frame at 30 and at 60 fps, so the cut lands on
a sample boundary exactly where the last frame ends."""

AUDIO_CHANNELS = (1, 2)

AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg")
"""What the GUI's file dialog offers first. Anything ffmpeg can decode is accepted,
including a video file, whose first audio track is used."""

FADE_SECONDS = 1
"""Length of the fade-out on audio that ends before the video does (user, 2026-10-14)."""

AAC_FRAME = 1024
"""Samples per AAC-LC frame. The encoder pads the last frame, so a track may read up to
this much longer than the samples it was given; the conformance check allows for it."""

_AAC_MAX_BITS_PER_CHANNEL = 6144
"""Per frame, ISO/IEC 14496-3. Above this the encoder clamps silently."""


class AudioError(RuntimeError):
    """The audio file cannot be used: missing, unreadable, or without an audio track."""


@dataclasses.dataclass(frozen=True, slots=True)
class AudioSpec:
    """Which file, and what the track in the delivery is encoded as."""

    path: pathlib.Path
    bitrate_kbps: int = DEFAULT_AUDIO_BITRATE_KBPS
    sample_rate: int = DEFAULT_AUDIO_RATE
    channels: int = DEFAULT_AUDIO_CHANNELS

    def __post_init__(self) -> None:
        if self.sample_rate not in AUDIO_RATES:
            raise ValueError(
                f"audio sample rate must be one of {AUDIO_RATES}, got {self.sample_rate}"
            )
        if self.channels not in AUDIO_CHANNELS:
            raise ValueError(f"audio channels must be one of {AUDIO_CHANNELS}, got {self.channels}")
        ceiling = _AAC_MAX_BITS_PER_CHANNEL * self.channels * self.sample_rate // AAC_FRAME // 1000
        if not 32 <= self.bitrate_kbps <= ceiling:
            raise ValueError(
                f"audio bitrate must be 32-{ceiling} kbps at {self.sample_rate} Hz with "
                f"{self.channels} channel(s), got {self.bitrate_kbps}"
            )

    @property
    def layout(self) -> str:
        return "stereo" if self.channels == 2 else "mono"

    def describe(self) -> str:
        return f"AAC-LC {self.sample_rate / 1000:g} kHz {self.layout} {self.bitrate_kbps} kbps"

    def codec_args(self) -> list[str]:
        """The encoder half of the mux; rate and layout are pinned by the filter graph."""
        return ["-c:a", "aac", "-profile:a", "aac_low", "-b:a", f"{self.bitrate_kbps}k"]


def _align(sample_rate: int) -> str:
    """Resample, then put the first sample at zero. Shared by the count and the mux."""
    return f"aresample={sample_rate},asetpts=PTS-STARTPTS"


@dataclasses.dataclass(frozen=True, slots=True)
class AudioPlan:
    """One audio file measured against one video: what gets cut, padded and faded."""

    spec: AudioSpec
    source_channels: int
    source_samples: int
    """The file's length at `spec.sample_rate`, from its first sample (after alignment)."""
    video_samples: int
    """The video's length in the same samples: frames x rate / fps."""
    frame_samples: int
    """One video frame, in samples."""

    @property
    def trimmed_samples(self) -> int:
        return max(self.source_samples - self.video_samples, 0)

    @property
    def silent_samples(self) -> int:
        return max(self.video_samples - self.source_samples, 0)

    @property
    def fades(self) -> bool:
        """Short by a frame or more. Less than that is rounding in somebody's export."""
        return self.silent_samples >= self.frame_samples

    @property
    def fade_samples(self) -> int:
        return min(FADE_SECONDS * self.spec.sample_rate, self.source_samples) if self.fades else 0

    def _channel_stage(self) -> str:
        if self.spec.channels == 2 and self.source_channels == 1:
            return "pan=stereo|c0=c0|c1=c0,"
        if self.spec.channels == 1 and self.source_channels == 2:
            return "pan=mono|c0=0.5*c0+0.5*c1,"
        return ""  # same count, or a multichannel source: aformat's standard downmix

    def filter_graph(self, source: str = "1:a:0", label: str = "aout") -> str:
        rate, end = self.spec.sample_rate, self.video_samples
        fade = (
            f",afade=t=out:ss={self.source_samples - self.fade_samples}:ns={self.fade_samples}"
            if self.fades
            else ""
        )
        return (
            f"[{source}]{_align(rate)},{self._channel_stage()}"
            f"aformat=sample_rates={rate}:channel_layouts={self.spec.layout},"
            f"atrim=end_sample={end}{fade},apad=whole_len={end}[{label}]"
        )

    def describe(self) -> str:
        rate = self.spec.sample_rate
        lengths = (
            f"{self.source_samples / rate:.2f} s of audio for {self.video_samples / rate:.2f} s "
            "of video"
        )
        if self.trimmed_samples:
            outcome = f"the last {self.trimmed_samples / rate:.2f} s cut"
        elif self.fades:
            outcome = (
                f"silent for the last {self.silent_samples / rate:.2f} s, the audio's last "
                f"{self.fade_samples / rate:.2f} s faded out"
            )
        else:
            outcome = "same length"
        return f"{self.spec.path.name} -> {self.spec.describe()}; {lengths}, {outcome}"


def _source_channels(tools: Tools, path: pathlib.Path) -> int:
    result = subprocess.run(
        [
            str(tools.ffprobe),
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,channels",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        errors="replace",
    )
    if result.returncode != 0:
        raise AudioError(f"cannot read {path}: {result.stderr.strip()}")
    streams = json.loads(result.stdout or "{}").get("streams", [])
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    if not audio:
        raise AudioError(f"{path} has no audio track")
    return int(audio[0].get("channels", 0))


def _decoded_samples(tools: Tools, path: pathlib.Path, sample_rate: int) -> int:
    """Samples the mux will see, counted by decoding to 8-bit mono (one byte each)."""
    command = [
        str(tools.ffmpeg),
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-af",
        _align(sample_rate),
        "-ac",
        "1",
        "-c:a",
        "pcm_u8",
        "-f",
        "u8",
        "-",
    ]
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        out, err = process.stdout, process.stderr
        assert out is not None and err is not None
        stderr: list[bytes] = []
        # A damaged file can make the decoder chatty enough to fill the stderr pipe
        # while this side is blocked reading stdout; drain it alongside.
        drain = threading.Thread(target=lambda: stderr.append(err.read()), daemon=True)
        drain.start()
        count = 0
        while chunk := out.read(1 << 20):
            count += len(chunk)
        code = process.wait()
        drain.join()
    if code != 0:
        message = b"".join(stderr).decode("utf-8", "replace").strip()
        raise AudioError(f"cannot decode the audio in {path}: {message}")
    return count


def plan(tools: Tools, spec: AudioSpec, frames: int, fps: int) -> AudioPlan:
    """Check the file and measure it against `frames` at `fps`. Raises :class:`AudioError`.

    Called before the warp plan is built, so a wrong path fails in a second rather than
    after the half hour of rendering it would otherwise be found at.
    """
    if not spec.path.is_file():
        raise AudioError(f"no such audio file: {spec.path}")
    channels = _source_channels(tools, spec.path)
    samples = _decoded_samples(tools, spec.path, spec.sample_rate)
    if samples == 0:
        raise AudioError(f"{spec.path} decodes to no audio")
    per_frame = fractions.Fraction(spec.sample_rate, fps)
    return AudioPlan(
        spec=spec,
        source_channels=channels,
        source_samples=samples,
        video_samples=round(per_frame * frames),
        frame_samples=round(per_frame),
    )
