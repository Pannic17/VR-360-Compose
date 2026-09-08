"""ffmpeg: locating it, building conformant encoder invocations, writing and probing MP4.

The delivery spec (AGENTS.md §5) pins container, codec, FourCC, profile, pixel format,
GOP structure and B-frames, and several of those fail *silently* if left to defaults:

* `libx265` picks Level 7.1 at 150-200 Mbps -- HEVC defines nothing above 6.2.
* `libx264 -preset ultrafast` emits Constrained Baseline, not High.
* `x265` has open-GOP on by default; the spec wants it closed.

So the level is computed from the standards' tables here, never left to the encoder, and
:func:`probe` reads back what was actually written so a test can assert every field.

ffmpeg and ffprobe are looked up next to the executable first (the distributed build
ships them alongside, AGENTS.md §7), then on PATH.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import numpy.typing as npt

U8 = npt.NDArray[np.uint8]

__all__ = [
    "BITRATES_KBPS",
    "SIZES",
    "EncodeSpec",
    "FfmpegNotFound",
    "SegmentWriter",
    "StreamInfo",
    "Tools",
    "concat",
    "find_tools",
    "select_level",
]

SIZES: dict[str, tuple[int, int]] = {"8k": (7680, 3840), "4k": (4096, 2048)}

BITRATES_KBPS: dict[str, tuple[int, int, int]] = {
    "8k": (200_000, 150_000, 100_000),
    "4k": (57_000, 43_000, 28_000),
}
"""(high, mid, low) in kbps. The 4k ladder holds the 8k ladder's bits per pixel
(pixel ratio 3.515625); dropping 4k from 200/150/100 Mbps cost 0.67-0.89 dB and made the
files 72% smaller (AGENTS.md §5)."""

LADDER = ("high", "mid", "low")

# H.264 Table A-1: level, MaxFS (macroblocks), MaxMBPS, MaxBR kbps (Main/Baseline).
# High profile multiplies MaxBR by 1.25.
H264_LEVELS: tuple[tuple[str, int, int, int], ...] = (
    ("4.0", 8_192, 245_760, 20_000),
    ("4.1", 8_192, 245_760, 50_000),
    ("4.2", 8_704, 522_240, 50_000),
    ("5.0", 22_080, 589_824, 135_000),
    ("5.1", 36_864, 983_040, 240_000),
    ("5.2", 36_864, 2_073_600, 240_000),
    ("6.0", 139_264, 4_177_920, 240_000),
    ("6.1", 139_264, 8_355_840, 480_000),
    ("6.2", 139_264, 16_711_680, 800_000),
)
H264_HIGH_BR_FACTOR = 1.25

# HEVC Tables A.6 / A.9: level, MaxLumaPs, MaxLumaSr, MaxBR Main tier, MaxBR High tier.
H265_LEVELS: tuple[tuple[str, int, int, int, int], ...] = (
    ("4.0", 2_228_224, 68_222_976, 12_000, 30_000),
    ("4.1", 2_228_224, 136_446_976, 20_000, 50_000),
    ("5.0", 8_912_896, 267_386_880, 25_000, 100_000),
    ("5.1", 8_912_896, 534_773_760, 40_000, 160_000),
    ("5.2", 8_912_896, 1_069_547_520, 60_000, 240_000),
    ("6.0", 35_651_584, 1_069_547_520, 60_000, 240_000),
    ("6.1", 35_651_584, 2_139_095_040, 120_000, 480_000),
    ("6.2", 35_651_584, 4_278_190_080, 240_000, 800_000),
)

COLOUR_ARGS = ("-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709")


class FfmpegNotFound(RuntimeError):
    """ffmpeg or ffprobe is not available, or lacks a required encoder."""


@dataclasses.dataclass(frozen=True, slots=True)
class Tools:
    ffmpeg: pathlib.Path
    ffprobe: pathlib.Path
    version: str
    encoders: frozenset[str]

    def require(self, *names: str) -> None:
        missing = [n for n in names if n not in self.encoders]
        if missing:
            raise FfmpegNotFound(
                f"{self.ffmpeg} lacks encoder(s) {missing}; a build with libx264 and libx265 "
                "is required (the distributed package ships one)"
            )


def _search_dirs() -> list[pathlib.Path]:
    """Next to the executable first. Frozen, that is the application directory."""
    if getattr(sys, "frozen", False):
        return [pathlib.Path(sys.executable).parent]
    here = pathlib.Path(__file__).resolve()
    return [pathlib.Path(sys.executable).parent, here.parents[2], here.parents[2] / "bin"]


def _locate(name: str) -> pathlib.Path | None:
    exe = f"{name}.exe" if sys.platform == "win32" else name
    for directory in _search_dirs():
        candidate = directory / exe
        if candidate.is_file():
            return candidate
    found = shutil.which(name)
    return pathlib.Path(found) if found else None


def find_tools() -> Tools:
    """Locate ffmpeg/ffprobe and read which encoders this build carries."""
    ffmpeg, ffprobe = _locate("ffmpeg"), _locate("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise FfmpegNotFound(
            "ffmpeg and ffprobe were not found next to the executable or on PATH. "
            "Put ffmpeg.exe and ffprobe.exe beside the application, or install ffmpeg."
        )
    version = subprocess.run(
        [str(ffmpeg), "-hide_banner", "-version"], capture_output=True, text=True
    ).stdout.splitlines()[0]
    listing = subprocess.run(
        [str(ffmpeg), "-hide_banner", "-encoders"], capture_output=True, text=True
    ).stdout
    encoders = frozenset(
        line.split()[1]
        for line in listing.splitlines()
        if line.startswith(" V") and len(line.split()) > 1
    )
    return Tools(ffmpeg=ffmpeg, ffprobe=ffprobe, version=version, encoders=encoders)


def select_level(width: int, height: int, codec: str, kbps: int, fps: int) -> tuple[str, str]:
    """Minimum conformant (level, tier) for a configuration.

    Picture size and sample rate fix the lowest possible level; the bitrate may push it
    up. For HEVC the search stays inside that level's family first (5.x for 4K, 6.x for
    8K): lowest level whose Main tier admits the bitrate, else lowest whose High tier
    does, and only then the next family. That is why 4K at 200 Mbps is 5.2 High rather
    than 6.2 Main. Lower is better throughout: a lower level asks less of the decoder.
    """
    if codec == "h264":
        macroblocks = -(-width // 16) * -(-height // 16)
        for level, max_fs, max_mbps, max_br in H264_LEVELS:
            if (
                macroblocks <= max_fs
                and macroblocks * fps <= max_mbps
                and kbps <= max_br * H264_HIGH_BR_FACTOR
            ):
                return level, "-"
        raise ValueError(f"no H.264 level admits {width}x{height} @ {fps} fps, {kbps} kbps")
    if codec == "h265":
        samples = width * height
        fits_picture = [
            entry for entry in H265_LEVELS if samples <= entry[1] and samples * fps <= entry[2]
        ]
        if not fits_picture:
            raise ValueError(f"no HEVC level admits {width}x{height} @ {fps} fps")
        # Group by level family (5.x, 6.x). Within the lowest family the picture allows:
        # lowest level whose Main tier admits the bitrate, else lowest whose High tier
        # does. Only then move up a family. Keeps 4K inside 5.x, which is what 4K
        # decoders advertise; a 6.x tag would exclude them for no gain.
        families: dict[str, list[tuple[str, int, int, int, int]]] = {}
        for entry in fits_picture:
            families.setdefault(entry[0].split(".")[0], []).append(entry)
        for family in sorted(families):
            for tier in ("main", "high"):
                for level, _ps, _sr, br_main, br_high in families[family]:
                    if kbps <= (br_main if tier == "main" else br_high):
                        return level, tier
        raise ValueError(f"no HEVC level admits {width}x{height} @ {fps} fps, {kbps} kbps")
    raise ValueError(f"unknown codec {codec!r}; expected 'h264' or 'h265'")


@dataclasses.dataclass(frozen=True, slots=True)
class EncodeSpec:
    """One delivery configuration, fully determined."""

    width: int
    height: int
    codec: str
    bitrate_kbps: int
    fps: int
    preset: str = "medium"
    full_range: bool = False
    deterministic: bool = False
    """Force a run-to-run reproducible bitstream, at a throughput cost.

    Measured on the stdin path. x264 with auto threads is reproducible for segments long
    enough to fill its frame-thread pipeline (60 frames at 4K and at 256x128 came out
    identical) but not for short ones: a 10-frame tail segment differed between runs, and
    so did `threads=8` on 128-row frames. `threads=1` is reproducible everywhere. x265 is
    never reproducible with frame threads; it needs `frame-threads=1:wpp=0`, measured at
    7.7x slower (10.8 -> 1.4 fps at 4K).

    Without the flag a resumed job is structurally identical to a clean one (same frames,
    GOPs, conformance) and, for x264, usually byte-identical too -- except for a short
    final segment. With it, byte-identical is guaranteed for both codecs.
    """

    def __post_init__(self) -> None:
        if self.codec not in ("h264", "h265"):
            raise ValueError(f"codec must be 'h264' or 'h265', got {self.codec!r}")
        if self.preset == "ultrafast":
            raise ValueError("ultrafast downgrades H.264 to Constrained Baseline; use medium+")
        if self.fps <= 0 or self.bitrate_kbps <= 0:
            raise ValueError("fps and bitrate must be positive")
        if self.width != 2 * self.height:
            raise ValueError(f"equirect output must be 2:1, got {self.width}x{self.height}")

    @classmethod
    def for_size(cls, size: str, codec: str, ladder: str, fps: int, **kwargs: object) -> EncodeSpec:
        width, height = SIZES[size]
        kbps = BITRATES_KBPS[size][LADDER.index(ladder)]
        return cls(width, height, codec, kbps, fps, **kwargs)  # type: ignore[arg-type]

    @property
    def gop(self) -> int:
        """Two seconds: 30 fps -> 60, 60 fps -> 120."""
        return self.fps * 2

    @property
    def level(self) -> str:
        return select_level(self.width, self.height, self.codec, self.bitrate_kbps, self.fps)[0]

    @property
    def tier(self) -> str:
        return select_level(self.width, self.height, self.codec, self.bitrate_kbps, self.fps)[1]

    @property
    def expected_profile(self) -> str:
        return "High" if self.codec == "h264" else "Main"

    @property
    def expected_tag(self) -> str:
        return "avc1" if self.codec == "h264" else "hvc1"

    def input_args(self) -> list[str]:
        """Raw RGB frames on stdin, exactly what :class:`SegmentWriter` feeds."""
        return [
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
        ]

    def video_args(self) -> list[str]:
        """The verified encoder invocation (AGENTS.md §5), audio excluded by `-an`."""
        gop = self.gop
        level, tier = self.level, self.tier
        # The RGB->YUV matrix is made explicit rather than left to swscale's default,
        # which is BT.601 and would silently disagree with the bt709 tags below.
        vf = (
            "scale=in_range=full"
            f":out_range={'full' if self.full_range else 'limited'}"
            ":out_color_matrix=bt709,format=yuv420p"
        )
        common = [
            "-vf",
            vf,
            "-pix_fmt",
            "yuv420p",
            *COLOUR_ARGS,
            "-color_range",
            "pc" if self.full_range else "tv",
            "-b:v",
            f"{self.bitrate_kbps}k",
            "-maxrate",
            f"{self.bitrate_kbps}k",
            "-bufsize",
            f"{2 * self.bitrate_kbps}k",
            "-an",
            "-movflags",
            "+faststart",
        ]
        structure = f"bframes=0:open-gop=0:keyint={gop}:min-keyint={gop}:scenecut=0"
        if self.codec == "h264":
            return [
                "-c:v",
                "libx264",
                "-preset",
                self.preset,
                "-profile:v",
                "high",
                "-level:v",
                level,
                "-x264-params",
                structure + (":threads=1" if self.deterministic else ""),
                "-tag:v",
                "avc1",
                *common,
            ]
        return [
            "-c:v",
            "libx265",
            "-preset",
            self.preset,
            "-profile:v",
            "main",
            "-x265-params",
            f"{structure}:level-idc={level}:high-tier={1 if tier == 'high' else 0}"
            + (":frame-threads=1:wpp=0" if self.deterministic else ""),
            "-tag:v",
            "hvc1",
            *common,
        ]

    def describe(self) -> str:
        tier = f" {self.tier} tier" if self.codec == "h265" else ""
        return (
            f"{self.codec} {self.width}x{self.height} @ {self.fps} fps, "
            f"{self.bitrate_kbps / 1000:g} Mbps, level {self.level}{tier}, GOP {self.gop}, "
            f"preset {self.preset}"
        )


@dataclasses.dataclass(frozen=True, slots=True)
class StreamInfo:
    """What ffprobe reports for the video stream -- every field the spec pins down."""

    codec: str
    profile: str
    level: str
    tag: str
    width: int
    height: int
    pix_fmt: str
    color_range: str
    color_space: str
    frames: int
    b_frames: int
    i_intervals: tuple[int, ...]
    bitrate_mbps: float
    audio_streams: int

    def conformance_problems(self, spec: EncodeSpec) -> list[str]:
        problems: list[str] = []
        if self.profile != spec.expected_profile:
            problems.append(f"profile {self.profile!r} != {spec.expected_profile!r}")
        if self.level != spec.level:
            problems.append(f"level {self.level} != {spec.level}")
        if self.tag != spec.expected_tag:
            problems.append(f"tag {self.tag!r} != {spec.expected_tag!r}")
        if (self.width, self.height) != (spec.width, spec.height):
            problems.append(f"size {self.width}x{self.height}")
        if self.pix_fmt != "yuv420p":
            problems.append(f"pix_fmt {self.pix_fmt}")
        if self.b_frames:
            problems.append(f"{self.b_frames} B-frames")
        if any(interval != spec.gop for interval in self.i_intervals):
            problems.append(f"I-frame intervals {self.i_intervals} != {spec.gop}")
        if self.audio_streams:
            problems.append(f"{self.audio_streams} audio stream(s); the spec has none")
        return problems


def probe(tools: Tools, path: pathlib.Path) -> StreamInfo:
    """Read back the stream as written. Slow-ish (decodes frame types); use on segments."""
    meta = json.loads(
        subprocess.run(
            [
                str(tools.ffprobe),
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_streams",
                "-show_format",
                str(path),
            ],
            capture_output=True,
            text=True,
        ).stdout
    )
    video = next(s for s in meta["streams"] if s["codec_type"] == "video")
    audio = sum(1 for s in meta["streams"] if s["codec_type"] == "audio")
    raw_level = video.get("level")
    level = ""
    if isinstance(raw_level, (int, float)):
        level = f"{raw_level / (30 if video['codec_name'] == 'hevc' else 10):.1f}"
    types = "".join(
        subprocess.run(
            [
                str(tools.ffprobe),
                "-v",
                "quiet",
                "-select_streams",
                "v:0",
                "-show_entries",
                "frame=pict_type",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
        ).stdout.split()
    ).replace(",", "")
    i_positions = [i for i, c in enumerate(types) if c == "I"]
    return StreamInfo(
        codec=str(video["codec_name"]),
        profile=str(video.get("profile", "")),
        level=level,
        tag=str(video.get("codec_tag_string", "")),
        width=int(video["width"]),
        height=int(video["height"]),
        pix_fmt=str(video.get("pix_fmt", "")),
        color_range=str(video.get("color_range", "")),
        color_space=str(video.get("color_space", "")),
        frames=len(types),
        b_frames=types.count("B"),
        i_intervals=tuple(sorted({b - a for a, b in itertools.pairwise(i_positions)})),
        bitrate_mbps=int(meta["format"].get("bit_rate", 0)) / 1e6,
        audio_streams=audio,
    )


class SegmentWriter:
    """Feeds RGB frames to one ffmpeg process producing one MP4 segment.

    Writes to ``<path>.part`` and renames on a clean exit, so a half-written segment is
    never mistaken for a finished one by the resume logic.
    """

    def __init__(self, tools: Tools, spec: EncodeSpec, path: pathlib.Path) -> None:
        self.spec = spec
        self.path = path
        self.partial = path.with_name(path.name + ".part")
        self.frames = 0
        self._frame_bytes = spec.width * spec.height * 3
        path.parent.mkdir(parents=True, exist_ok=True)
        self.partial.unlink(missing_ok=True)
        self._process: subprocess.Popen[bytes] = subprocess.Popen(
            [
                str(tools.ffmpeg),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                *spec.input_args(),
                *spec.video_args(),
                # No creation_time or encoder-version tags: with these, a segment written
                # twice from the same frames is byte-identical (x264; x265 if deterministic).
                "-fflags",
                "+bitexact",
                "-flags",
                "+bitexact",
                "-f",
                "mp4",
                str(self.partial),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: U8) -> None:
        if frame.shape != (self.spec.height, self.spec.width, 3) or frame.dtype != np.uint8:
            raise ValueError(
                f"frame must be ({self.spec.height}, {self.spec.width}, 3) uint8, "
                f"got {frame.shape} {frame.dtype}"
            )
        assert self._process.stdin is not None
        try:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError:
            raise RuntimeError(f"ffmpeg exited early:\n{self._stderr()}") from None
        self.frames += 1

    def _stderr(self) -> str:
        assert self._process.stderr is not None
        data: bytes = self._process.stderr.read()
        return data.decode(errors="replace").strip()

    def close(self) -> pathlib.Path:
        """Finish the segment. Raises if ffmpeg failed; returns the final path."""
        assert self._process.stdin is not None
        self._process.stdin.close()
        code = self._process.wait()
        err = self._stderr()
        if code != 0:
            self.partial.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg exited with {code} on {self.partial.name}:\n{err}")
        self.partial.replace(self.path)
        return self.path

    def abort(self) -> None:
        """Kill ffmpeg and remove the partial file. For cancellation."""
        if self._process.poll() is None:
            self._process.kill()
            self._process.wait()
        self.partial.unlink(missing_ok=True)

    def __enter__(self) -> SegmentWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def concat(tools: Tools, segments: list[pathlib.Path], output: pathlib.Path) -> pathlib.Path:
    """Join segments losslessly. Valid because every segment starts on a closed GOP."""
    if not segments:
        raise ValueError("nothing to concatenate")
    listing = output.with_name(output.name + ".concat.txt")
    listing.write_text(
        "".join(f"file '{p.resolve().as_posix()}'\n" for p in segments), encoding="utf-8"
    )
    partial = output.with_name(output.name + ".part")
    try:
        result = subprocess.run(
            [
                str(tools.ffmpeg),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(listing),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                "-fflags",
                "+bitexact",
                "-flags",
                "+bitexact",
                "-f",
                "mp4",
                str(partial),
            ],
            capture_output=True,
            text=True,
        )
    finally:
        listing.unlink(missing_ok=True)
    if result.returncode != 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"concat failed:\n{result.stderr.strip()}")
    partial.replace(output)
    return output
