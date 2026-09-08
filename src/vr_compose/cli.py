"""Command-line entry point.

The GUI in P6 must be a thin shell over exactly this code, so every operation lives in a
library module and this file only parses arguments and prints. Nothing here hardcodes a
source path: `--source` is optional and discovery fills it in (AGENTS.md §2, constraint 3).
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from collections.abc import Sequence

from tqdm import tqdm

from vr_compose import __version__, encode, io, pipeline, verify
from vr_compose import source as source_mod
from vr_compose.rig import Rig, UnknownRigError, rig_for
from vr_compose.stitch import DEFAULT_BAND_ROWS, DEFAULT_SAMPLER, SAMPLERS, stitch_frame
from vr_compose.warp import DEFAULT_THREADS


def _master_size(
    chosen: source_mod.SourceSet, rig: Rig, size: str, stitch_at: str
) -> tuple[int, int] | None:
    """The size to stitch, or None to stitch straight at the delivery size.

    `native` renders the panorama at the source's own sampling density and lets swscale
    resample it down inside the encode -- measured 2.52 dB closer to the master than
    warping straight to a smaller grid (ROADMAP P3). It is also what makes a 16K render
    delivered at 8K work: the master follows the tiles, the delivery follows the spec.

    None comes back when there is nothing to resample (native density *is* the delivery
    size, the usual 1920-tile 8K case) or when the tiles are unusable, which
    `run_sequence` reports properly.
    """
    if stitch_at == "delivery":
        return None
    tile = chosen.tile_size
    if tile is None:
        return None
    width = rig.native_width(tile)
    if width <= encode.SIZES[size][0]:
        return None
    return width, width // 2


def _resolve_source(explicit: pathlib.Path | None) -> source_mod.SourceSet:
    """Pick exactly one source set, or exit with something actionable."""
    found, searched = source_mod.discover(explicit)
    if not found:
        print("no usable source set found. searched:", file=sys.stderr)
        for path in searched[:12]:
            print(f"  {path}", file=sys.stderr)
        if len(searched) > 12:
            print(f"  ... and {len(searched) - 12} more", file=sys.stderr)
        if explicit is not None:
            for candidate in source_mod.scan(explicit):
                print(f"\nrejected {explicit} / stem {candidate.stem!r}:", file=sys.stderr)
                for problem in candidate.problems:
                    print(f"  - {problem}", file=sys.stderr)
        print(
            "\nA source directory needs at least two `CameraN` directories whose PNGs are\n"
            "named `<stem>.<frame>.png`, either directly inside or one level below.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if len(found) > 1:
        stems = ", ".join(repr(s.stem) for s in found)
        print(
            f"{len(found)} source sets in {found[0].root} (stems: {stems}).\n"
            "Choose one with --stem; processing an arbitrary first match would risk the "
            "wrong scene.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return found[0]


def _select_stem(candidates: list[source_mod.SourceSet], stem: str | None) -> source_mod.SourceSet:
    if stem is None:
        return candidates[0]
    for candidate in candidates:
        if candidate.stem == stem:
            return candidate
    available = ", ".join(repr(c.stem) for c in candidates)
    raise SystemExit(f"no stem {stem!r} here; available: {available}")


def cmd_discover(args: argparse.Namespace) -> int:
    found, searched = source_mod.discover(args.source)
    print(f"searched {len(searched)} location(s)\n")
    if not found:
        for path in searched[:12]:
            print(f"  {path}")
        rejected = source_mod.scan(args.source) if args.source else []
        for candidate in rejected:
            print(f"\nrejected stem {candidate.stem!r}:")
            for problem in candidate.problems:
                print(f"  - {problem}")
        return 1
    for candidate in found:
        print(candidate.describe())
        try:
            rig = rig_for(candidate.camera_count)
            print(
                f"rig        : {rig.name}  fov={rig.fov_deg:g}deg  "
                f"{len(rig.unique_indices)} distinct of {rig.file_count} files"
            )
            if rig.duplicate_indices:
                print(f"redundant  : cameras {list(rig.duplicate_indices)} duplicate other views")
        except UnknownRigError as exc:
            print(f"rig        : UNKNOWN -- {exc}")
        print()
    return 0


def cmd_frame(args: argparse.Namespace) -> int:
    candidates = [s for s in source_mod.scan(args.source) if s.usable] if args.source else []
    chosen = _select_stem(candidates, args.stem) if candidates else _resolve_source(args.source)

    try:
        rig = rig_for(chosen.camera_count)
    except UnknownRigError as exc:
        raise SystemExit(str(exc)) from None

    frames = chosen.frames
    frame = args.frame if args.frame is not None else frames[0]
    if frame not in frames:
        near = ", ".join(str(f) for f in frames[:5])
        raise SystemExit(
            f"frame {frame} is not present in every camera. "
            f"{len(frames)} frames available, starting {near}..."
        )

    tile = chosen.tile_size
    if tile is None:
        raise SystemExit("source tiles are not square or not uniform; cannot stitch")
    width = args.width or rig.native_width(tile)

    indices = list(rig.unique_indices)
    print(f"source     : {chosen.root}  stem {chosen.stem!r}")
    print(f"rig        : {rig.name}, reading {len(indices)} of {rig.file_count} files")
    started = time.time()
    tiles = io.load_tiles(chosen, frame, indices, workers=args.decode_workers)
    decoded = time.time() - started

    result = stitch_frame(tiles, rig, width, band_rows=args.band_rows, sampler=args.sampler)
    elapsed = time.time() - started
    print(
        f"stitched   : frame {frame} at {width}x{width // 2} in {elapsed:.1f} s "
        f"(decode {decoded:.1f} s), overlap {result.overlap_fraction:.1%}, "
        f"up to {result.max_contributors} tiles"
    )
    print(f"wrap seam  : {verify.wrap_seam_error(result.image):.2f} / 255")
    report = verify.agreement(result.stats)
    print(report.report(args.sampler))

    if args.out is not None:
        size = io.write_png(args.out, result.image, compress_level=args.compress_level)
        print(f"wrote      : {args.out}  ({size / 2**20:.1f} MiB)")
    return 0 if report.passed else 1


DELIVERY_ONLY = (
    "size",
    "codec",
    "bitrate",
    "fps",
    "segment_gops",
    "deterministic",
    "encoder_threads",
    "keep_segments",
    "stitch_at",
)
"""`sequence` options that only mean something when an encoder is involved."""

MASTER_ONLY = ("compress_level", "write_workers")
"""...and the ones that only mean something when PNG files are."""


def _reject_inapplicable(args: argparse.Namespace, names: tuple[str, ...], why: str) -> None:
    """Refuse options that cannot apply, rather than ignoring them (AGENTS.md §10).

    Compares against the parser's own defaults, so passing a default value explicitly is
    not treated as asking for anything.
    """
    given = [name for name in names if getattr(args, name) != args.option_defaults[name]]
    if given:
        flags = ", ".join("--" + name.replace("_", "-") for name in given)
        raise SystemExit(f"{flags} {why}")


def cmd_master(
    args: argparse.Namespace, chosen: source_mod.SourceSet, rig: Rig, frames: list[int]
) -> int:
    """`sequence --out-format png`: a lossless PNG master per frame.

    Masters are always at the source's native density -- the sampling density the tiles
    were rendered at, `Rig.native_width` -- because a master is the thing other sizes are
    derived *from*. For a one-off at some other width, `frame --width` is the command.
    """
    _reject_inapplicable(
        args,
        DELIVERY_ONLY,
        "only apply to --out-format mp4. A PNG master has no codec, bitrate, frame rate "
        "or delivery size to choose: it is the source's own density, losslessly.",
    )
    tile = chosen.tile_size
    if tile is None:
        raise SystemExit("source tiles are not square or not uniform; cannot stitch")
    width = rig.native_width(tile)
    default_name = f"{chosen.stem}.{frames[0]}-{frames[-1]}.{width}x{width // 2}.masters"
    try:
        job = pipeline.MasterJob(
            source=chosen,
            rig=rig,
            frames=tuple(frames),
            directory=args.out or pipeline.default_output_dir() / default_name,
            width=width,
            compress_level=args.compress_level,
            decode_workers=args.decode_workers,
            warp_threads=args.warp_threads,
            sampler=args.sampler,
            write_workers=args.write_workers,
            stats_every=args.stats_every,
            resume=not args.no_resume,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    pending = job.pending_frames()
    print(f"source     : {chosen.root}  stem {chosen.stem!r}")
    print(f"rig        : {rig.name}, reading {len(rig.unique_indices)} of {rig.file_count} files")
    print(f"frames     : {frames[0]}..{frames[-1]} ({len(frames)}), {len(pending)} to render")
    print(
        f"master     : {width}x{width // 2} PNG, compress_level {job.compress_level}, "
        f"sampler {job.sampler}"
    )
    print(f"output     : {job.directory}")

    bar = tqdm(
        total=len(frames),
        unit="frame",
        desc="mastering",
        dynamic_ncols=True,
        disable=args.no_bar,
        mininterval=1.0,
    )
    bar.update(len(frames) - len(pending))

    def on_progress(report: pipeline.Progress) -> None:
        bar.n = report.done
        bar.set_postfix_str(f"{report.seconds_per_frame:.2f} s/frame", refresh=False)
        bar.refresh()

    def on_log(message: str) -> None:
        tqdm.write(f"           {message}")

    try:
        summary = pipeline.run_master(job, progress=on_progress, log=on_log)
    except pipeline.GeometryGateFailed as exc:
        raise SystemExit(
            f"geometry gate FAILED -- stopping before wasting the run:\n{exc}"
        ) from None
    except (pipeline.Cancelled, KeyboardInterrupt) as exc:
        detail = f" ({exc})" if isinstance(exc, pipeline.Cancelled) else ""
        print(
            f"\ninterrupted{detail}; finished masters are kept. Re-run the same command to resume.",
            file=sys.stderr,
        )
        return 130
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
    finally:
        bar.close()

    print(f"\ndone       : {summary.written} master(s) -> {summary.directory}")
    if summary.written:
        print(
            f"throughput : {summary.seconds_per_frame:.2f} s/frame, "
            f"{summary.bytes_per_frame / 2**20:.1f} MiB/frame "
            f"({summary.bytes_written / 2**30:.2f} GiB written)"
        )
        print(f"stages     : {summary.stages.report()}")
    if summary.skipped:
        print(f"resumed    : {summary.skipped} master(s) were already present")
    if summary.gate_reports:
        worst = max(report.median for report in summary.gate_reports)
        gates = len(summary.gate_reports)
        print(f"geometry   : {gates} sampled frame(s) PASS, worst median {worst:.2f}")
    if summary.peak.measured:
        print(f"memory     : {summary.peak.report()}")
    for warning in summary.warnings:
        print(f"WARNING    : {warning}", file=sys.stderr)
    return 0


def cmd_sequence(args: argparse.Namespace) -> int:
    candidates = [s for s in source_mod.scan(args.source) if s.usable] if args.source else []
    chosen = _select_stem(candidates, args.stem) if candidates else _resolve_source(args.source)
    try:
        rig = rig_for(chosen.camera_count)
    except UnknownRigError as exc:
        raise SystemExit(str(exc)) from None
    try:
        frames = pipeline.parse_frames(args.frames, chosen.frames)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    if args.out_format == "exr":
        raise SystemExit(
            "EXR masters are not implemented: the source is 8-bit PNG, so an EXR would "
            "carry no more information than the PNG master does. The interface is "
            "reserved for a 16-bit source (AGENTS.md §11, item 5). Use --out-format png."
        )
    if args.out_format == "png":
        return cmd_master(args, chosen, rig, frames)
    _reject_inapplicable(
        args, MASTER_ONLY, "only apply to --out-format png; an MP4 is not a PNG sequence."
    )

    try:
        spec = encode.EncodeSpec.for_size(
            args.size,
            args.codec,
            args.bitrate,
            args.fps,
            preset=args.preset,
            deterministic=args.deterministic,
            encoder_threads=args.encoder_threads,
            master=_master_size(chosen, rig, args.size, args.stitch_at),
        )
        job = pipeline.SequenceJob(
            source=chosen,
            rig=rig,
            spec=spec,
            frames=tuple(frames),
            output=args.out
            or pipeline.default_output_dir() / pipeline.default_output_name(chosen, spec, frames),
            segment_gops=args.segment_gops,
            decode_workers=args.decode_workers,
            warp_threads=args.warp_threads,
            sampler=args.sampler,
            stats_every=args.stats_every,
            resume=not args.no_resume,
            keep_segments=args.keep_segments,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    print(f"source     : {chosen.root}  stem {chosen.stem!r}")
    print(f"rig        : {rig.name}, reading {len(rig.unique_indices)} of {rig.file_count} files")
    print(
        f"frames     : {frames[0]}..{frames[-1]} ({len(frames)}), "
        f"{len(job.segments())} segment(s) of up to {job.segment_frames}"
    )
    print(f"sampler    : {args.sampler}")
    print(f"encode     : {spec.describe()}")
    print(f"output     : {job.output}")

    bar = tqdm(
        total=len(frames),
        unit="frame",
        desc="stitching",
        dynamic_ncols=True,
        disable=args.no_bar,
        mininterval=1.0,
    )

    def on_progress(report: pipeline.Progress) -> None:
        bar.n = report.done
        speedup = (
            pipeline.PREVIOUS_PIPELINE_SECONDS_PER_FRAME / report.seconds_per_frame
            if report.seconds_per_frame
            else 0.0
        )
        bar.set_postfix_str(
            f"seg {report.segment + 1}/{report.segments}  "
            f"{report.seconds_per_frame:.2f} s/frame  {speedup:.1f}x vs 46.85",
            refresh=False,
        )
        bar.refresh()

    def on_log(message: str) -> None:
        # tqdm.write keeps the bar intact; a bare print would smear it
        tqdm.write(f"           {message}")

    try:
        summary = pipeline.run_sequence(job, progress=on_progress, log=on_log)
    except pipeline.GeometryGateFailed as exc:
        raise SystemExit(
            f"geometry gate FAILED -- stopping before wasting the run:\n{exc}"
        ) from None
    # Cancelled is a RuntimeError, so it has to be caught before the failure branch below.
    except (pipeline.Cancelled, KeyboardInterrupt) as exc:
        detail = f" ({exc})" if isinstance(exc, pipeline.Cancelled) else ""
        print(
            f"\ninterrupted{detail}; finished segments are kept. "
            "Re-run the same command to resume.",
            file=sys.stderr,
        )
        return 130
    except (encode.FfmpegNotFound, RuntimeError) as exc:
        raise SystemExit(str(exc)) from None
    finally:
        bar.close()

    size_mib = summary.output.stat().st_size / 2**20
    print(f"\ndone       : {summary.frames} frames -> {summary.output} ({size_mib:.0f} MiB)")
    if summary.encoded:
        speedup = pipeline.PREVIOUS_PIPELINE_SECONDS_PER_FRAME / summary.seconds_per_frame
        print(
            f"throughput : {summary.seconds_per_frame:.2f} s/frame over {summary.encoded} "
            f"encoded frame(s) ({speedup:.1f}x vs the previous pipeline's 46.85)"
        )
    if summary.encoded:
        print(f"stages     : {summary.stages.report()}")
    if summary.peak.measured:
        print(f"memory     : {summary.peak.report()}")
    if summary.skipped_segments:
        print(f"resumed    : {summary.skipped_segments} segment(s) were already complete")
    gates = summary.gate_reports
    if gates:
        worst = max(report.median for report in gates)
        print(f"geometry   : {len(gates)} sampled frame(s) PASS, worst median {worst:.2f}")
    stream = summary.stream
    print(
        f"stream     : {stream.codec} {stream.profile} L{stream.level} {stream.tag} "
        f"{stream.width}x{stream.height} {stream.pix_fmt} B={stream.b_frames} "
        f"I-gap={list(stream.i_intervals)} {stream.bitrate_mbps:.1f} Mbps"
    )
    for warning in summary.warnings:
        print(f"WARNING    : {warning}", file=sys.stderr)
    if summary.problems:
        print("CONFORMANCE PROBLEMS:")
        for problem in summary.problems:
            print(f"  - {problem}")
        return 1
    print("conformance: OK")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vr-compose", description="VR 360 composition toolkit")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--source",
        type=pathlib.Path,
        default=None,
        help="source directory; omitted means search next to the executable",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    discover = sub.add_parser("discover", help="report the detected source sets and rig")
    discover.set_defaults(func=cmd_discover)

    frame = sub.add_parser("frame", help="stitch a single frame and check its geometry")
    frame.add_argument("--frame", type=int, default=None, help="default: the first shared frame")
    frame.add_argument(
        "--stem", default=None, help="which scene (file stem), when a directory has several"
    )
    frame.add_argument(
        "--width",
        type=int,
        default=None,
        help="output width (2:1); default: the source's native density (4x the tile)",
    )
    frame.add_argument("--out", type=pathlib.Path, default=None, help="write the panorama here")
    frame.add_argument("--compress-level", type=int, default=6, choices=range(10))
    frame.add_argument("--band-rows", type=int, default=DEFAULT_BAND_ROWS)
    frame.add_argument("--decode-workers", type=int, default=8)
    frame.add_argument(
        "--sampler",
        choices=list(SAMPLERS),
        default=DEFAULT_SAMPLER,
        help="how a tile is read at a fractional position; 'nearest' is the P1 baseline",
    )
    frame.set_defaults(func=cmd_frame)

    sequence = sub.add_parser("sequence", help="stitch a frame range straight into a delivery MP4")
    sequence.add_argument("--frames", default="all", help="'all', '1656-2433', '1656-', or a list")
    sequence.add_argument(
        "--stem", default=None, help="which scene (file stem), when a directory has several"
    )
    sequence.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help="output .mp4; default: <stem>.<first>-<last>.<WxH>.<codec>.mp4 beside the program",
    )
    sequence.add_argument(
        "--out-format",
        choices=("mp4", "png", "exr"),
        default="mp4",
        help="'mp4' (default) is the delivery path; 'png' writes a lossless master per "
        "frame at the source's native density, for quality work and archiving",
    )
    sequence.add_argument("--size", choices=list(encode.SIZES), default="8k")
    sequence.add_argument("--codec", choices=["h264", "h265"], default="h264")
    sequence.add_argument(
        "--bitrate",
        choices=list(encode.LADDER),
        default="high",
        help="rung of the per-size ladder: 8k 200/150/100 Mbps, 4k 57/43/28 Mbps",
    )
    sequence.add_argument("--fps", type=int, default=30, choices=(30, 60))
    sequence.add_argument("--preset", default="medium", help="x264/x265 preset; never ultrafast")
    sequence.add_argument(
        "--stitch-at",
        choices=("native", "delivery"),
        default="native",
        help="'native' (default) stitches at the source's own density (4x the tile) and "
        "lets the encoder resample down; 'delivery' warps straight to the output size -- "
        "faster, but it point-samples an oversampled source and aliases. Preview only",
    )
    sequence.add_argument(
        "--deterministic",
        action="store_true",
        help="byte-reproducible x265 (frame-threads=1:wpp=0, ~8x slower); x264 already is",
    )
    sequence.add_argument("--segment-gops", type=int, default=5, help="GOPs per resumable segment")
    sequence.add_argument(
        "--decode-workers",
        type=int,
        default=4,
        help="threads decoding the next frame; more than 4 slows the warp (GIL contention)",
    )
    sequence.add_argument("--warp-threads", type=int, default=DEFAULT_THREADS)
    sequence.add_argument(
        "--sampler",
        choices=list(SAMPLERS),
        default=DEFAULT_SAMPLER,
        help="'bilinear' (default) interpolates; 'nearest' is the P1 baseline, faster but "
        "it places every sample up to half a pixel out",
    )
    sequence.add_argument(
        "--encoder-threads",
        type=int,
        default=None,
        help="cap the encoder's threads to lower its memory; unset (default) is fastest "
        "and commits about 21 GiB at 8K, 16 commits about 8 GiB. Footprint only -- it "
        "does not make the output reproducible",
    )
    sequence.add_argument("--stats-every", type=int, default=100, help="geometry gate cadence")
    sequence.add_argument("--no-bar", action="store_true", help="plain log lines, no tqdm bar")
    sequence.add_argument("--no-resume", action="store_true", help="re-encode finished segments")
    sequence.add_argument("--keep-segments", action="store_true")
    sequence.add_argument(
        "--compress-level",
        type=int,
        default=1,
        choices=range(10),
        help="PNG master zlib level; 1 costs 0.68 s a frame against 6's 2.77 s for ~10%% "
        "more bytes. The pixels are identical either way (--out-format png only)",
    )
    sequence.add_argument(
        "--write-workers",
        type=int,
        default=1,
        help="PNG encodes in flight; 1 measured fastest at 8K -- a second worker hides "
        "the write but slows the warp (--out-format png only)",
    )
    # Attached so `_reject_inapplicable` can tell "asked for" from "left alone".
    sequence.set_defaults(
        func=cmd_sequence,
        option_defaults={
            name: sequence.get_default(name) for name in (*DELIVERY_ONLY, *MASTER_ONLY)
        },
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result
