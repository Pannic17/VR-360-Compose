"""Can we win back any of the 4.3 dB that 4:2:0 costs, by doing the conversion ourselves?

**Answered, 8K, 2026-09-08: no -- and most of that 4.3 dB was never in the file.**
The delivery colour path is unchanged as a result. Two findings, both reproducible with
one command; the numbers live in AGENTS.md section 5.

**1. The headline penalty was mostly the measuring instrument.** The recorded baseline
(43.21 dB, mean DC shift -1.23) was measured by letting ffmpeg convert `yuv420p` straight
to `rgb24`, and that step is not neutral. With the encoder taken out, on real 8K frames:

    4:4:4, exact BT.709 inverse   52.07 dB   DC +0.00
    4:4:4, via swscale            52.07 dB   DC +0.00   <- validates the exact inverse
    4:2:0, exact BT.709 inverse   49.70 dB   DC -0.02
    4:2:0, via swscale            44.25 dB   DC -1.18

The decimation costs 2.38 dB and does not darken anything. ffmpeg's 4:2:0 -> RGB step
costs a further 5.45 dB and *all* of the darkening -- a flat offset at every brightness
(binned by reference luma: -1.26 / -1.17 / -1.12 / -1.21 / -1.19). Its chroma *upsampling*
is fine, and measurably better than nearest; it is the matrix step on the 4:2:0 path.
Consequence for the project: every PSNR figure measured through that path understates the
file by about 3.7 dB. Comparisons between configurations still hold -- the bias is one
constant -- but the absolute values are not delivery quality (AGENTS.md section 9,
metric C). Whether a viewer sees the 1.2/255 depends on their player's own YUV -> RGB.

**2. None of the four variants is worth a second colour path.** Doing the conversion
ourselves in gamma space wins +0.10 dB, which does not pay for feeding ffmpeg `yuv420p`
instead of `rgb24` and giving up the verified reproducible path. Linear light does
*nothing* (-0.02 dB against the gamma box -- the second time this project has measured
linear light and rejected it; P4 rejected it for blending). The "correct" chroma sample
position is *worse* (-0.26 dB), because a downsampler is only optimal against the
matching upsampler, and the upsampler belongs to the player: measured over all four
pairings at 4k, today's box downsample plus an interpolating upsample is the best of
them. Dither has no bias left to fix, so it only adds noise (-1.05 dB).

The variants, each encoded with the same x265 invocation the delivery uses:

    swscale (baseline)    what the pipeline does today: rgb24 in, swscale converts
    ours / gamma box      our own conversion, 2x2 box on chroma in gamma space (control:
                          this should land on the baseline, and if it does not, the
                          comparison below is measuring our bug rather than the physics)
    ours / linear box     2x2 box in *linear light*, then re-encoded
    ours / linear sited   linear light with the correct 4:2:0 sample position:
                          [1 2 1]/4 horizontally, co-sited with the even luma column,
                          and a 2-tap vertical average (chroma_loc = left)
    ours / linear dither  the above plus triangular dither on the 8-bit quantisation

Run it:

    python tools/chroma_probe.py --frames 8
    python tools/chroma_probe.py --size 4k --frames 8 --codec h265

The verdict rule was set before the measurement (ROADMAP P7 item 4): keep a variant only
if it beats the baseline measurably. A tenth of a dB is not worth a second colour path.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import subprocess
import sys
import time

import numpy as np
import numpy.typing as npt
from PIL import Image

from vr_compose import encode as vr_encode

Image.MAX_IMAGE_PIXELS = None

F32 = npt.NDArray[np.float32]
U8 = npt.NDArray[np.uint8]

DEFAULT_ROOT = pathlib.Path("E:/22")
OUTPUT_SUBDIR = "FinishTaskOutput"
OUTPUT_STEM = "MonoEye.L_Cathedral"

# BT.709 luma weights and the Pb/Pr denominators (Rec. ITU-R BT.709-6).
KR, KG, KB = 0.2126, 0.7152, 0.0722
PB_DENOM = 2.0 * (1.0 - KB)  # 1.8556
PR_DENOM = 2.0 * (1.0 - KR)  # 1.5748


def reference_path(root: pathlib.Path, frame: int) -> pathlib.Path:
    return root / OUTPUT_SUBDIR / f"{OUTPUT_STEM}.{frame:04d}.png"


def srgb_to_linear(code: F32) -> F32:
    """0..255 sRGB code -> linear. The source is assumed sRGB, as elsewhere in `tools/`."""
    c = code / 255.0
    return np.asarray(
        np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4), dtype=np.float32
    )


def linear_to_srgb(linear: F32) -> F32:
    """Linear -> 0..255 sRGB code. Inverse of :func:`srgb_to_linear`."""
    c = np.clip(linear, 0.0, 1.0)
    low = c * 12.92
    high = 1.055 * np.power(np.maximum(c, 1e-8), 1.0 / 2.4) - 0.055
    return np.asarray(np.where(c <= 0.0031308, low, high) * 255.0, dtype=np.float32)


def _quantise(value: F32, rng: np.random.Generator | None) -> U8:
    """8-bit, rounded -- or dithered, which trades a little noise for an unbiased mean.

    Triangular (TPDF) noise of +-1 LSB is the standard choice: rectangular dither leaves a
    signal-dependent noise floor, and rounding alone puts a fixed bias into every value
    that sits between two codes.
    """
    if rng is None:
        return np.asarray(np.clip(np.rint(value), 0, 255), dtype=np.uint8)
    noise = rng.random(value.shape, dtype=np.float32) - rng.random(value.shape, dtype=np.float32)
    return np.asarray(np.clip(np.rint(value + noise), 0, 255), dtype=np.uint8)


def _downsample_box(plane: F32) -> F32:
    """2x2 average -- the position swscale uses by default, and the wrong one for 4:2:0."""
    height, width = plane.shape
    return np.asarray(
        plane.reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3)), dtype=np.float32
    )


def _downsample_sited(plane: F32) -> F32:
    """The 4:2:0 position H.264/HEVC actually signal by default (`chroma_loc = left`).

    Horizontally the chroma sample is *co-sited* with the even luma sample, so the filter
    is [1 2 1]/4 centred on it -- not a box straddling two columns, which shifts the whole
    chroma plane half a pixel to the right. Vertically it sits between the two rows, so
    there a 2-tap average is correct.
    """
    height, width = plane.shape
    left = np.roll(plane, 1, axis=1)  # wraps: an equirect panorama is periodic in x
    right = np.roll(plane, -1, axis=1)
    horizontal = (left + 2.0 * plane + right) * 0.25
    even = horizontal[:, 0::2]
    return np.asarray(even.reshape(height // 2, 2, width // 2).mean(axis=1), dtype=np.float32)


@dataclasses.dataclass(frozen=True, slots=True)
class Variant:
    label: str
    linear: bool = False
    sited: bool = False
    dither: bool = False

    @property
    def ours(self) -> bool:
        return self.label != "swscale (baseline)"


VARIANTS = (
    Variant("swscale (baseline)"),
    Variant("ours / gamma box"),
    Variant("ours / linear box", linear=True),
    Variant("ours / linear sited", linear=True, sited=True),
    Variant("ours / linear dither", linear=True, sited=True, dither=True),
)


def to_yuv420p(rgb: U8, variant: Variant, rng: np.random.Generator | None) -> bytes:
    """Our own limited-range BT.709 4:2:0, as three planes ready for ffmpeg's stdin.

    Luma is always computed per pixel from the gamma-encoded values -- that is what Y' in
    BT.709 *is*, and moving it to linear light would produce a different signal, not a
    better one. Only the chroma decimation is up for discussion.
    """
    gamma = rgb.astype(np.float32)
    r, g, b = gamma[:, :, 0], gamma[:, :, 1], gamma[:, :, 2]
    luma = (KR * r + KG * g + KB * b) / 255.0
    y = _quantise(16.0 + 219.0 * luma, rng)

    if variant.linear:
        # Average radiance, then re-encode: the average of two codes is not the code of
        # the average light, and on saturated content the difference does not cancel.
        linear = srgb_to_linear(gamma)
        down = _downsample_sited if variant.sited else _downsample_box
        chroma_rgb = np.stack([linear_to_srgb(down(linear[:, :, c])) for c in range(3)], axis=-1)
    else:
        down = _downsample_sited if variant.sited else _downsample_box
        chroma_rgb = np.stack([down(gamma[:, :, c]) for c in range(3)], axis=-1)

    cr_, cg_, cb_ = chroma_rgb[:, :, 0], chroma_rgb[:, :, 1], chroma_rgb[:, :, 2]
    chroma_luma = (KR * cr_ + KG * cg_ + KB * cb_) / 255.0
    pb = (cb_ / 255.0 - chroma_luma) / PB_DENOM
    pr = (cr_ / 255.0 - chroma_luma) / PR_DENOM
    u = _quantise(128.0 + 224.0 * pb, rng)
    v = _quantise(128.0 + 224.0 * pr, rng)
    return y.tobytes() + u.tobytes() + v.tobytes()


def encode_variant(
    tools: vr_encode.Tools,
    spec: vr_encode.EncodeSpec,
    frames: list[U8],
    variant: Variant,
    out: pathlib.Path,
    seed: int,
) -> float:
    """Encode `frames` through one variant. Returns wall seconds."""
    started = time.time()
    if not variant.ours:
        # The pipeline's own path, unchanged: raw RGB in, swscale does the conversion.
        with vr_encode.SegmentWriter(tools, spec, out) as writer:
            for frame in frames:
                writer.write(frame)
        return time.time() - started

    rng = np.random.default_rng(seed) if variant.dither else None
    process = subprocess.Popen(
        [
            str(tools.ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-s",
            f"{spec.width}x{spec.height}",
            "-r",
            str(spec.fps),
            # The planes are already limited-range BT.709 4:2:0. Tagging the *input* as
            # such is what keeps ffmpeg from "helpfully" converting them again; there is
            # no -vf at all on this path.
            "-color_range",
            "tv",
            "-colorspace",
            "bt709",
            "-i",
            "-",
            *_video_args_without_filter(spec),
            "-fflags",
            "+bitexact",
            "-flags",
            "+bitexact",
            "-f",
            "mp4",
            str(out),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    try:
        for frame in frames:
            process.stdin.write(to_yuv420p(frame, variant, rng))
        process.stdin.close()
    except OSError:
        pass
    code = process.wait()
    if code != 0:
        assert process.stderr is not None
        raise SystemExit(f"ffmpeg failed on {variant.label}: {process.stderr.read().decode()}")
    return time.time() - started


def _video_args_without_filter(spec: vr_encode.EncodeSpec) -> list[str]:
    """The delivery encoder arguments, minus the RGB->YUV filter we are replacing.

    Everything else -- codec, level, GOP structure, bitrate, colour tags -- comes from
    :class:`EncodeSpec`, so a variant differs from the delivery in exactly one respect.
    """
    args = spec.video_args()
    out: list[str] = []
    skip = False
    for arg in args:
        if skip:
            skip = False
            continue
        if arg == "-vf":
            skip = True
            continue
        out.append(arg)
    return out


def yuv_to_rgb(y: F32, u: F32, v: F32) -> F32:
    """The exact BT.709 limited-range inverse, in float. All three planes full size.

    This is the reference reconstruction the numbers below are measured against, and it
    is validated against ffmpeg itself: on a 4:4:4 file the two agree to the last code
    (see :func:`round_trip`, `4:4:4` row). It exists because ffmpeg's *4:2:0* path does
    not agree -- see :func:`decompose`.
    """
    luma = (y - 16.0) / 219.0
    pb = (u - 128.0) / 224.0
    pr = (v - 128.0) / 224.0
    red = (luma + PR_DENOM * pr) * 255.0
    blue = (luma + PB_DENOM * pb) * 255.0
    green = (luma * 255.0 - KR * red - KB * blue) / KG
    return np.asarray(np.stack([red, green, blue], axis=-1), dtype=np.float32)


def _quantised_rgb(value: F32) -> U8:
    return np.asarray(np.clip(np.rint(value), 0, 255), dtype=np.uint8)


def _raw(tools: vr_encode.Tools, args: list[str], stdin: bytes | None = None) -> bytes:
    result = subprocess.run(
        [str(tools.ffmpeg), "-hide_banner", "-v", "error", *args, "-f", "rawvideo", "-"],
        input=stdin,
        capture_output=True,
    )
    return result.stdout


def decode_swscale(tools: vr_encode.Tools, path: pathlib.Path, width: int, height: int) -> list[U8]:
    """Decode to RGB entirely inside ffmpeg -- the path every earlier measurement used."""
    buf = np.frombuffer(_raw(tools, ["-i", str(path), "-pix_fmt", "rgb24"]), np.uint8)
    per = width * height * 3
    return [
        np.asarray(buf[i * per : (i + 1) * per].reshape(height, width, 3), dtype=np.uint8)
        for i in range(len(buf) // per)
    ]


def decode_exact(tools: vr_encode.Tools, path: pathlib.Path, width: int, height: int) -> list[U8]:
    """Decode with ffmpeg, but do the YUV->RGB step here, with :func:`yuv_to_rgb`.

    ffmpeg still decodes the bitstream and upsamples the chroma (its upsampler measures
    *better* than a nearest-neighbour one, so this is not a way of flattering ourselves);
    only the matrix and range conversion move out of swscale, because that is where the
    flat -1.2/255 offset lives.
    """
    buf = np.frombuffer(_raw(tools, ["-i", str(path), "-pix_fmt", "yuv444p"]), np.uint8)
    per = width * height
    out: list[U8] = []
    for i in range(len(buf) // (3 * per)):
        base = i * 3 * per
        planes = [
            buf[base + p * per : base + (p + 1) * per].reshape(height, width).astype(np.float32)
            for p in range(3)
        ]
        out.append(_quantised_rgb(yuv_to_rgb(*planes)))
    return out


def compare(decoded: list[U8], reference: list[U8]) -> dict[str, float]:
    mses: list[float] = []
    maes: list[float] = []
    dcs: list[float] = []
    for got, want in zip(decoded, reference, strict=False):
        delta = got.astype(np.int16) - want.astype(np.int16)
        mses.append(float((delta.astype(np.float64) ** 2).mean()))
        maes.append(float(np.abs(delta).mean()))
        dcs.append(float(delta.mean()))
    if not mses:
        return {"psnr": float("nan"), "mae": float("nan"), "dc": float("nan")}
    mse = float(np.mean(mses))
    return {
        "psnr": 10 * np.log10(255**2 / mse) if mse else float("inf"),
        "mae": float(np.mean(maes)),
        "dc": float(np.mean(dcs)),
    }


def load_frames(root: pathlib.Path, first: int, count: int, size: str) -> list[U8]:
    """The lossless master frames, at the delivery size (Lanczos, as the encoder would)."""
    width, height = vr_encode.SIZES[size]
    out: list[U8] = []
    for i in range(count):
        path = reference_path(root, first + i)
        if not path.is_file():
            raise SystemExit(f"missing reference frame: {path}")
        image = Image.open(path).convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        out.append(np.asarray(image, dtype=np.uint8))
    return out


def to_yuv444p(rgb: U8) -> bytes:
    """Limited-range BT.709 with no decimation at all: the ceiling to measure against."""
    gamma = rgb.astype(np.float32)
    r, g, b = gamma[:, :, 0], gamma[:, :, 1], gamma[:, :, 2]
    luma = (KR * r + KG * g + KB * b) / 255.0
    y = _quantise(16.0 + 219.0 * luma, None)
    u = _quantise(128.0 + 224.0 * (b / 255.0 - luma) / PB_DENOM, None)
    v = _quantise(128.0 + 224.0 * (r / 255.0 - luma) / PR_DENOM, None)
    return y.tobytes() + u.tobytes() + v.tobytes()


def round_trip(
    tools: vr_encode.Tools,
    frames: list[U8],
    size: str,
    *,
    planar: str,
    make: object,
    exact: bool,
) -> dict[str, float]:
    """RGB -> planar YUV -> RGB with **no encoder**, so only the colour path is measured.

    `exact` chooses who does the final matrix: :func:`yuv_to_rgb` here, or swscale. When
    `planar` is 4:4:4 there is no chroma resampling for anyone to do, so the two must
    agree -- and they do, which is what licenses using the exact inverse everywhere else.
    """
    width, height = vr_encode.SIZES[size]
    decoded: list[U8] = []
    for frame in frames:
        raw = make(frame)  # type: ignore[operator]
        # Exact: let ffmpeg upsample the chroma to 4:4:4 and stop there, so the matrix and
        # range conversion are ours. A 4:4:4 input passes through untouched.
        target = "yuv444p" if exact else "rgb24"
        filters = (
            [] if exact else ["-vf", "scale=in_range=limited:out_range=full:in_color_matrix=bt709"]
        )
        buf = np.frombuffer(
            _raw(
                tools,
                [
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    planar,
                    "-s",
                    f"{width}x{height}",
                    "-color_range",
                    "tv",
                    "-colorspace",
                    "bt709",
                    "-i",
                    "-",
                    *filters,
                    "-pix_fmt",
                    target,
                ],
                raw,
            ),
            np.uint8,
        )
        if exact:
            per = width * height
            planes = [
                buf[p * per : (p + 1) * per].reshape(height, width).astype(np.float32)
                for p in range(3)
            ]
            decoded.append(_quantised_rgb(yuv_to_rgb(*planes)))
        else:
            decoded.append(
                np.asarray(buf[: width * height * 3].reshape(height, width, 3), dtype=np.uint8)
            )
    return compare(decoded, frames)


def decompose(tools: vr_encode.Tools, frames: list[U8], size: str) -> None:
    """Where the 4:2:0 penalty actually goes, with the encoder taken out of the loop.

    This is the part worth reading. The 4.3 dB and the -1.23 DC shift recorded in
    AGENTS.md section 5 were both measured by letting ffmpeg convert yuv420p straight to
    rgb24, and that conversion is not neutral: it puts a flat offset of about -1.2/255
    into every sample, at every brightness. On a 4:4:4 file the same path is exact, so the
    offset is specific to ffmpeg's 4:2:0 -> RGB conversion, not to the decimation, and not
    to the bytes we deliver.
    """
    box = Variant("ours / gamma box")
    rows = [
        ("4:4:4, exact inverse", round_trip(tools, frames, size, planar="yuv444p",
                                            make=to_yuv444p, exact=True)),
        ("4:4:4, via swscale", round_trip(tools, frames, size, planar="yuv444p",
                                          make=to_yuv444p, exact=False)),
        ("4:2:0, exact inverse", round_trip(tools, frames, size, planar="yuv420p",
                                            make=lambda f: to_yuv420p(f, box, None), exact=True)),
        ("4:2:0, via swscale", round_trip(tools, frames, size, planar="yuv420p",
                                          make=lambda f: to_yuv420p(f, box, None), exact=False)),
    ]  # fmt: skip
    print("\nno encoder, colour path only:")
    print(f"{'path':<24} {'PSNR':>8} {'MAE':>6} {'DC shift':>9}")
    for label, stats in rows:
        print(f"{label:<24} {stats['psnr']:8.2f} {stats['mae']:6.2f} {stats['dc']:+9.2f}")
    exact444, sws444, exact420, sws420 = (row[1]["psnr"] for row in rows)
    print(
        f"\n  4:4:4 exact vs swscale : {sws444 - exact444:+.2f} dB "
        f"-- agreement here is what validates the exact inverse"
    )
    print(f"  the decimation costs   : {exact420 - exact444:+.2f} dB, DC {rows[2][1]['dc']:+.2f}")
    print(
        f"  ffmpeg's 4:2:0 -> RGB  : {sws420 - exact420:+.2f} dB, "
        f"DC {rows[3][1]['dc'] - rows[2][1]['dc']:+.2f} -- measurement, not delivery"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    parser.add_argument("--root", type=pathlib.Path, default=DEFAULT_ROOT)
    parser.add_argument("--work", type=pathlib.Path, default=pathlib.Path("E:/vrc_chroma"))
    parser.add_argument("--size", choices=list(vr_encode.SIZES), default="8k")
    parser.add_argument("--codec", choices=["h264", "h265"], default="h265")
    parser.add_argument("--bitrate", choices=list(vr_encode.LADDER), default="high")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument(
        "--first",
        type=int,
        default=1,
        help="first reference frame; the previous pipeline's finished panoramas are "
        "numbered 1..1655 in FinishTaskOutput (the 778 remaining *source* frames, 1656+, "
        "have no lossless panorama to compare against)",
    )
    parser.add_argument("--seed", type=int, default=11, help="dither seed")
    parser.add_argument(
        "--no-decompose",
        action="store_true",
        help="skip the encoder-free breakdown of where the 4:2:0 penalty actually goes",
    )
    args = parser.parse_args(argv)

    tools = vr_encode.find_tools()
    tools.require("libx264", "libx265")
    args.work.mkdir(parents=True, exist_ok=True)
    width, height = vr_encode.SIZES[args.size]
    spec = vr_encode.EncodeSpec.for_size(args.size, args.codec, args.bitrate, args.fps)

    print(f"{spec.describe()}, {args.frames} frames from {args.first}")
    print(f"reference  : {args.root / OUTPUT_SUBDIR} (lossless PNG, resampled to {width}x{height})")
    print(f"ffmpeg     : {tools.version.splitlines()[0]}\n")
    frames = load_frames(args.root, args.first, args.frames, args.size)

    print("                       --- exact BT.709 inverse ---   -- via swscale --")
    print(
        f"{'variant':<22} {'PSNR':>8} {'vs base':>8} {'MAE':>6} {'DC':>7} "
        f"{'PSNR':>9} {'DC':>7} {'encode':>8}"
    )
    baseline: float | None = None
    for variant in VARIANTS:
        out = args.work / f"chroma_{variant.label.split('/')[-1].strip().replace(' ', '_')}.mp4"
        seconds = encode_variant(tools, spec, frames, variant, out, args.seed)
        exact = compare(decode_exact(tools, out, width, height), frames)
        sws = compare(decode_swscale(tools, out, width, height), frames)
        if baseline is None:
            baseline = exact["psnr"]
        print(
            f"{variant.label:<22} {exact['psnr']:8.2f} {exact['psnr'] - baseline:+8.2f} "
            f"{exact['mae']:6.2f} {exact['dc']:+7.2f} {sws['psnr']:9.2f} {sws['dc']:+7.2f} "
            f"{seconds:7.1f}s"
        )
        out.unlink(missing_ok=True)

    if not args.no_decompose:
        decompose(tools, frames, args.size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
