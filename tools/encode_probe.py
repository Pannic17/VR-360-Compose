"""Delivery-spec encoder probe: conformance matrix, rate-quality curve, chroma diagnosis.

Encodes short clips from the reference PNG master and verifies every field the delivery
spec pins down (AGENTS.md §5), because several of these fail *silently*:

* ``libx264 -preset ultrafast`` emits Constrained Baseline, not High, even with
  ``-profile:v high``. Only ffprobe catches it.
* ``libx265`` with auto level emits Level 7.1 at 150/200 Mbps. HEVC defines nothing above
  6.2, so no decoder will take it. Levels must be pinned.
* ``hevc_nvenc`` auto-selects Main tier L6.1, whose 120 Mbps ceiling the 200 Mbps option
  exceeds -- a non-conformant stream that encodes and plays fine locally.

Subcommands:

    python tools/encode_probe.py matrix          # spec conformance, all size/codec/bitrate
    python tools/encode_probe.py rate            # PSNR vs the lossless master
    python tools/encode_probe.py chroma          # how much 4:2:0 costs, and why
    python tools/encode_probe.py speed           # encode throughput, full-length estimate
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import pathlib
import shutil
import subprocess
import sys
import time

import numpy as np
import numpy.typing as npt
from PIL import Image

DEFAULT_ROOT = pathlib.Path("E:/22")
OUTPUT_SUBDIR = "FinishTaskOutput"
OUTPUT_STEM = "MonoEye.L_Cathedral"


def reference_path(root: pathlib.Path, frame: int) -> pathlib.Path:
    return root / OUTPUT_SUBDIR / f"{OUTPUT_STEM}.{frame:04d}.png"


Image.MAX_IMAGE_PIXELS = None

U8 = npt.NDArray[np.uint8]

SIZES: dict[str, tuple[int, int]] = {"8k": (7680, 3840), "4k": (4096, 2048)}

BITRATES_KBPS: dict[str, tuple[int, ...]] = {
    "8k": (200_000, 150_000, 100_000),
    "4k": (57_000, 43_000, 28_000),
}
"""Delivery bitrates in kbps. "200k" means 200_000 kbps = 200 Mbps (confirmed).

The 4k ladder is the 8k ladder scaled by the pixel ratio
(7680*3840) / (4096*2048) = 3.515625, holding bit/pixel constant, then rounded down to a
round number. Dropping 4k from 200/150/100 to 57/43/28 Mbps costs about 0.6 dB PSNR
(measured, AGENTS.md §5) and cuts the file from 2.1 GB to 0.6 GB.
"""

EXPECTED_PROFILE = {"h264": "High", "h265": "Main"}
EXPECTED_TAG = {"h264": "avc1", "h265": "hvc1"}

COLOUR_ARGS = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]

# --- Level tables, so any bitrate can be given its minimum conformant level ---------
#
# H.264 (ITU-T H.264 Table A-1): level, MaxFS in macroblocks, MaxMBPS, MaxBR in kbps.
# MaxBR listed for Main/Baseline; High profile multiplies it by 1.25 (cpbBrNalFactor).
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

# HEVC (ITU-T H.265 Tables A.6/A.9): level, MaxLumaPs, MaxLumaSr, MaxBR Main, MaxBR High.
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


def select_level(size: str, codec: str, kbps: int, fps: int) -> tuple[str, str]:
    """Minimum conformant (level, tier) for one configuration.

    Lower is better: a lower level asks less of the decoder. Frame size, sample rate and
    bitrate each impose a floor, so take the highest of the three. For HEVC, Main tier is
    preferred and High tier used only when no defined Main-tier level admits the bitrate.

    Never leave the level to the encoder: `libx265` picks Level 7.1 at these bitrates,
    which HEVC does not define, and `hevc_nvenc` picks a Main tier level whose bitrate
    ceiling the stream then exceeds. Both fail silently (AGENTS.md §2).
    """
    width, height = SIZES[size]
    if codec == "h264":
        macroblocks = (width // 16) * (height // 16)
        for level, max_fs, max_mbps, max_br in H264_LEVELS:
            if (
                macroblocks <= max_fs
                and macroblocks * fps <= max_mbps
                and kbps <= max_br * H264_HIGH_BR_FACTOR
            ):
                return level, "-"
        raise ValueError(f"no H.264 level admits {size} @ {fps} fps, {kbps} kbps")
    samples = width * height
    for tier_name in ("main", "high"):
        for level, max_ps, max_sr, max_br_main, max_br_high in H265_LEVELS:
            max_br = max_br_main if tier_name == "main" else max_br_high
            if samples <= max_ps and samples * fps <= max_sr and kbps <= max_br:
                return level, tier_name
    raise ValueError(f"no HEVC level admits {size} @ {fps} fps, {kbps} kbps")


def _kbps_list(value: str) -> list[int]:
    return [int(part) for part in value.replace(" ", "").split(",") if part]


def require_ffmpeg() -> None:
    for exe in ("ffmpeg", "ffprobe"):
        if shutil.which(exe) is None:
            raise SystemExit(f"{exe} not on PATH -- the delivery path needs ffmpeg")


def gop_for(fps: int) -> int:
    """2-second GOP: 30 fps -> 60, 60 fps -> 120."""
    return fps * 2


def video_args(
    size: str,
    codec: str,
    kbps: int,
    fps: int,
    *,
    full_range: bool = False,
    pix_fmt: str = "yuv420p",
    preset: str = "medium",
) -> list[str]:
    """The verified encoder invocation for one delivery configuration."""
    if preset == "ultrafast":
        raise ValueError("ultrafast downgrades H.264 to Constrained Baseline; use medium+")
    level, tier = select_level(size, codec, kbps, fps)
    gop = gop_for(fps)
    common = [
        *COLOUR_ARGS,
        "-pix_fmt",
        pix_fmt,
        "-color_range",
        "pc" if full_range else "tv",
        "-b:v",
        f"{kbps}k",
        "-maxrate",
        f"{kbps}k",
        "-bufsize",
        f"{2 * kbps}k",
        "-movflags",
        "+faststart",
    ]
    if codec == "h264":
        return [
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-profile:v",
            "high",
            "-level:v",
            level,
            "-x264-params",
            f"bframes=0:open-gop=0:keyint={gop}:min-keyint={gop}:scenecut=0",
            "-tag:v",
            "avc1",
            *common,
        ]
    profile = "main444-8" if pix_fmt == "yuv444p" else "main"
    return [
        "-c:v",
        "libx265",
        "-preset",
        preset,
        "-profile:v",
        profile,
        "-x265-params",
        f"bframes=0:open-gop=0:keyint={gop}:min-keyint={gop}:scenecut=0"
        f":level-idc={level}:high-tier={1 if tier == 'high' else 0}",
        "-tag:v",
        "hvc1",
        *common,
    ]


def scale_args(size: str) -> list[str]:
    width, height = SIZES[size]
    if (width, height) == SIZES["8k"]:
        return []
    return ["-vf", f"scale={width}:{height}:flags=lanczos"]


def encode(
    source_pattern: str,
    out: pathlib.Path,
    frames: int,
    fps: int,
    size: str,
    extra: list[str],
    vf: list[str] | None = None,
) -> tuple[bool, str, float]:
    out.unlink(missing_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(fps),
        "-start_number",
        "1",
        "-i",
        source_pattern,
        "-frames:v",
        str(frames),
        *(vf if vf is not None else scale_args(size)),
        *extra,
        str(out),
    ]
    started = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - started
    if result.returncode != 0 or not out.exists():
        return False, " ".join(result.stderr.split())[:110], elapsed
    return True, "", elapsed


@dataclasses.dataclass(frozen=True, slots=True)
class StreamInfo:
    """Every field the delivery spec pins down, as ffprobe reports it."""

    codec: str
    profile: str
    level: str
    tag: str
    width: int
    height: int
    pix_fmt: str
    color_range: str
    b_frames: int
    pict_types: str
    i_intervals: list[int]
    bitrate_mbps: float


def probe(path: pathlib.Path) -> StreamInfo:
    meta = json.loads(
        subprocess.run(
            [
                "ffprobe",
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
    stream = next(s for s in meta["streams"] if s["codec_type"] == "video")
    raw_level = stream.get("level")
    level = ""
    if isinstance(raw_level, (int, float)):
        # AVC carries level*10 in the SPS, HEVC level*30.
        level = f"{raw_level / (30 if stream['codec_name'] == 'hevc' else 10):.1f}"
    types = "".join(
        subprocess.run(
            [
                "ffprobe",
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
        codec=str(stream["codec_name"]),
        profile=str(stream.get("profile", "")),
        level=level,
        tag=str(stream.get("codec_tag_string", "")),
        width=int(stream["width"]),
        height=int(stream["height"]),
        pix_fmt=str(stream["pix_fmt"]),
        color_range=str(stream.get("color_range", "")),
        b_frames=types.count("B"),
        pict_types=types,
        i_intervals=sorted({b - a for a, b in itertools.pairwise(i_positions)}),
        bitrate_mbps=int(meta["format"]["bit_rate"]) / 1e6,
    )


def decode_frames(path: pathlib.Path, indices: list[int], width: int, height: int) -> U8:
    select = "+".join(f"eq(n\\,{n})" for n in indices)
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            f"select='{select}'",
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "-",
        ],
        capture_output=True,
    )
    buf = np.frombuffer(result.stdout, np.uint8)
    n = len(buf) // (width * height * 3)
    return np.asarray(buf[: n * width * height * 3].reshape(n, height, width, 3), np.uint8)


def master_frames(root: pathlib.Path, indices: list[int], size: str) -> list[npt.NDArray[np.int16]]:
    """The lossless reference, downscaled with the same filter the encoder would use."""
    width, height = SIZES[size]
    out: list[npt.NDArray[np.int16]] = []
    for i in indices:
        path = reference_path(root, i + 1)
        if size == "8k":
            out.append(np.asarray(Image.open(path).convert("RGB"), np.int16))
            continue
        result = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-vf",
                f"scale={width}:{height}:flags=lanczos",
                "-pix_fmt",
                "rgb24",
                "-f",
                "rawvideo",
                "-",
            ],
            capture_output=True,
        )
        out.append(
            np.asarray(
                np.frombuffer(result.stdout, np.uint8)[: width * height * 3].reshape(
                    height, width, 3
                ),
                np.int16,
            )
        )
    return out


def compare(decoded: U8, refs: list[npt.NDArray[np.int16]]) -> dict[str, float]:
    mses, maes, dcs = [], [], []
    for k in range(min(len(decoded), len(refs))):
        delta = decoded[k].astype(np.int16) - refs[k]
        mses.append(float((delta.astype(np.float64) ** 2).mean()))
        maes.append(float(np.abs(delta).mean()))
        dcs.append(float(delta.mean()))
    mse = float(np.mean(mses))
    return {
        "psnr": 10 * np.log10(255**2 / mse) if mse else float("inf"),
        "mae": float(np.mean(maes)),
        "dc": float(np.mean(dcs)),
    }


def cmd_matrix(args: argparse.Namespace) -> int:
    pattern = str(args.root / OUTPUT_SUBDIR / f"{OUTPUT_STEM}.%04d.png")
    print(
        f"{args.frames} frames per config, preset={args.preset}, fps={args.fps}, "
        f"GOP={gop_for(args.fps)}\n"
    )
    header = (
        f"{'size':<5} {'codec':<6} {'Mbps':>5} {'profile':<8} {'level':>5} {'tier':<5} "
        f"{'tag':<5} {'resolution':<11} {'pix_fmt':<8} {'B':>2} {'I-gap':>6} "
        f"{'actual':>8}  verdict"
    )
    print(header)
    failures = 0
    for size in SIZES:
        for codec in ("h264", "h265"):
            for kbps in args.bitrates or BITRATES_KBPS[size]:
                level, tier = select_level(size, codec, kbps, args.fps)
                out = args.work / f"matrix_{size}_{codec}_{kbps // 1000}.mp4"
                ok, err, _ = encode(
                    pattern,
                    out,
                    args.frames,
                    args.fps,
                    size,
                    video_args(size, codec, kbps, args.fps, preset=args.preset),
                )
                if not ok:
                    print(f"{size:<5} {codec:<6} {kbps // 1000:>5} ENCODE FAILED: {err}")
                    failures += 1
                    continue
                info = probe(out)
                problems: list[str] = []
                if info.profile != EXPECTED_PROFILE[codec]:
                    problems.append(f"profile={info.profile}")
                if info.level != level:
                    problems.append(f"level={info.level}")
                if info.tag != EXPECTED_TAG[codec]:
                    problems.append(f"tag={info.tag}")
                if info.pix_fmt != "yuv420p":
                    problems.append(f"pix_fmt={info.pix_fmt}")
                if info.b_frames:
                    problems.append(f"B={info.b_frames}")
                if (info.width, info.height) != SIZES[size]:
                    problems.append(f"{info.width}x{info.height}")
                # a clip shorter than one GOP has a single I-frame, hence no interval
                if info.i_intervals not in ([], [gop_for(args.fps)]):
                    problems.append(f"I-interval={info.i_intervals}")
                verdict = "OK" if not problems else "FAIL: " + " ".join(problems)
                failures += bool(problems)
                print(
                    f"{size:<5} {codec:<6} {kbps // 1000:>5} {info.profile:<8} "
                    f"{info.level:>5} {tier:<5} {info.tag:<5} "
                    f"{info.width}x{info.height:<6} {info.pix_fmt:<8} "
                    f"{info.b_frames:>2} {info.i_intervals!s:>7} "
                    f"{info.bitrate_mbps:7.1f}M  {verdict}"
                )
                out.unlink(missing_ok=True)
    print(f"\n{'all configurations conformant' if not failures else f'{failures} FAILURES'}")
    print(
        "Note: 'actual' is not a conformance check on a clip this short -- with a "
        "2-second VBV buffer a short clip legitimately overshoots the average. Check "
        "the average bitrate on the full-length render instead."
    )
    return 1 if failures else 0


def cmd_rate(args: argparse.Namespace) -> int:
    pattern = str(args.root / OUTPUT_SUBDIR / f"{OUTPUT_STEM}.%04d.png")
    width, height = SIZES[args.size]
    picks = [0, args.frames // 3, args.frames - 1]
    refs = master_frames(args.root, picks, args.size)
    pixels_per_second = width * height * args.fps
    ladder = args.bitrates or BITRATES_KBPS[args.size]
    print(
        f"{args.size} {width}x{height}, {args.frames} frames, PSNR vs the lossless master"
        f"{' (same lanczos downscale)' if args.size != '8k' else ''}\n"
    )
    print(
        f"{'codec':<6} {'target':>7} {'actual':>9} {'bit/px':>7} {'PSNR':>7} {'MAE':>6} "
        f"{'DC':>6} {'81.1 s file':>12}"
    )
    for codec in args.codecs:
        for kbps in ladder:
            out = args.work / f"rate_{args.size}_{codec}_{kbps // 1000}.mp4"
            ok, err, _ = encode(
                pattern,
                out,
                args.frames,
                args.fps,
                args.size,
                video_args(args.size, codec, kbps, args.fps),
            )
            if not ok:
                print(f"{codec:<6} {kbps // 1000:>6}M FAILED: {err}")
                continue
            stats = compare(decode_frames(out, picks, width, height), refs)
            actual = out.stat().st_size * 8 / 1e6 / (args.frames / args.fps)
            print(
                f"{codec:<6} {kbps // 1000:>6}M {actual:8.1f}M "
                f"{kbps * 1000 / pixels_per_second:7.3f} {stats['psnr']:7.2f} "
                f"{stats['mae']:6.2f} {stats['dc']:+6.2f} {actual * 81.1 / 8:9.0f} MB"
            )
            out.unlink(missing_ok=True)
    print(
        "\nNote: PSNR is nearly flat across these bitrates. The ceiling is 4:2:0, "
        "not quantisation -- run the `chroma` subcommand."
    )
    return 0


def cmd_chroma(args: argparse.Namespace) -> int:
    """Split the mp4 loss into colour-pipeline vs compression."""
    pattern = str(args.root / OUTPUT_SUBDIR / f"{OUTPUT_STEM}.%04d.png")
    width, height = SIZES[args.size]
    picks = [0, max(1, args.frames // 2)]
    refs = master_frames(args.root, picks, args.size)
    kbps = (args.bitrates or BITRATES_KBPS[args.size])[0]
    variants: list[tuple[str, dict[str, object], list[str]]] = [
        ("4:2:0 limited (spec)", {}, []),
        (
            "4:2:0 limited +accurate_rnd",
            {},
            ["-sws_flags", "+accurate_rnd+full_chroma_int+full_chroma_inp"],
        ),
        ("4:2:0 full range", {"full_range": True}, []),
        ("4:4:4 (off-spec, diagnostic)", {"pix_fmt": "yuv444p"}, []),
    ]
    print(f"{args.size} {width}x{height}, h265, {kbps // 1000} Mbps, {args.frames} frames\n")
    print(f"{'variant':<30} {'PSNR':>7} {'MAE':>6} {'DC shift':>9}")
    for label, kwargs, pre in variants:
        out = args.work / f"chroma_{label.split()[0].replace(':', '')}_{len(pre)}.mp4"
        extra = video_args(args.size, "h265", kbps, args.fps, **kwargs)  # type: ignore[arg-type]
        vf = scale_args(args.size)
        if kwargs.get("full_range"):
            vf = [
                "-vf",
                ",".join(
                    ([f"scale={width}:{height}:flags=lanczos"] if args.size != "8k" else [])
                    + ["scale=in_range=full:out_range=full"]
                ),
            ]
        out.unlink(missing_ok=True)
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            *pre,
            "-framerate",
            str(args.fps),
            "-start_number",
            "1",
            "-i",
            pattern,
            "-frames:v",
            str(args.frames),
            *vf,
            *extra,
            str(out),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"{label:<30} FAILED {' '.join(result.stderr.split())[:60]}")
            continue
        stats = compare(decode_frames(out, picks, width, height), refs)
        print(f"{label:<30} {stats['psnr']:7.2f} {stats['mae']:6.2f} {stats['dc']:+9.2f}")
        out.unlink(missing_ok=True)
    print(
        "\n4:2:0 is the quality ceiling, and ffmpeg's own switches do not recover it. "
        "Doing the RGB->YUV 4:2:0 conversion ourselves in linear light is the P4 "
        "experiment (AGENTS.md §5)."
    )
    return 0


def cmd_speed(args: argparse.Namespace) -> int:
    pattern = str(args.root / OUTPUT_SUBDIR / f"{OUTPUT_STEM}.%04d.png")
    total = args.total_frames
    print(
        f"{args.frames} frames per config (includes PNG decode), extrapolated to {total} frames\n"
    )
    print(f"{'size':<5} {'codec':<6} {'Mbps':>5} {'elapsed':>8} {'fps':>6} {f'{total} frames':>14}")
    for size in args.sizes:
        for codec in args.codecs:
            kbps = (args.bitrates or BITRATES_KBPS[size])[0]
            out = args.work / f"speed_{size}_{codec}.mp4"
            ok, err, elapsed = encode(
                pattern, out, args.frames, args.fps, size, video_args(size, codec, kbps, args.fps)
            )
            if not ok:
                print(f"{size:<5} {codec:<6} {kbps // 1000:>5} FAILED: {err}")
                continue
            fps = args.frames / elapsed
            print(
                f"{size:<5} {codec:<6} {kbps // 1000:>5} {elapsed:7.1f}s {fps:6.2f} "
                f"{total / fps / 60:11.1f} min"
            )
            out.unlink(missing_ok=True)
    print(
        "\nEncoding is not the bottleneck. Pipe the stitcher straight into ffmpeg and "
        "the 0.68-2.77 s/frame PNG encode disappears entirely (AGENTS.md §6)."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    require_ffmpeg()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=pathlib.Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--work",
        type=pathlib.Path,
        default=pathlib.Path("."),
        help="scratch directory for the test clips (they are deleted)",
    )
    parser.add_argument("--fps", type=int, default=30, choices=(30, 60))
    # comma-separated, not nargs="+": a greedy list would swallow the subcommand name
    parser.add_argument(
        "--bitrates",
        type=_kbps_list,
        default=None,
        metavar="KBPS[,KBPS...]",
        help="override the per-size ladder (default: 8k 200000,150000,100000; "
        "4k 57000,43000,28000)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("matrix", help="spec conformance across every configuration")
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--preset", default="medium")
    p.set_defaults(func=cmd_matrix)

    p = sub.add_parser("rate", help="rate-quality curve vs the lossless master")
    p.add_argument("--frames", type=int, default=90)
    p.add_argument("--size", choices=list(SIZES), default="8k")
    p.add_argument("--codecs", nargs="+", default=["h265", "h264"])
    p.set_defaults(func=cmd_rate)

    p = sub.add_parser("chroma", help="how much 4:2:0 costs, and whether ffmpeg can fix it")
    p.add_argument("--frames", type=int, default=12)
    p.add_argument("--size", choices=list(SIZES), default="8k")
    p.set_defaults(func=cmd_chroma)

    p = sub.add_parser("speed", help="encode throughput and full-length estimate")
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--sizes", nargs="+", default=list(SIZES))
    p.add_argument("--codecs", nargs="+", default=["h264", "h265"])
    p.add_argument("--total-frames", type=int, default=2433)
    p.set_defaults(func=cmd_speed)

    args = parser.parse_args(argv)
    args.work.mkdir(parents=True, exist_ok=True)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
