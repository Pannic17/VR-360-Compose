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
from vr_compose.stitch import DEFAULT_BAND_ROWS, stitch_frame
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

    result = stitch_frame(tiles, rig, width, band_rows=args.band_rows)
    elapsed = time.time() - started
    print(
        f"stitched   : frame {frame} at {width}x{width // 2} in {elapsed:.1f} s "
        f"(decode {decoded:.1f} s), overlap {result.overlap_fraction:.1%}, "
        f"up to {result.max_contributors} tiles"
    )
    print(f"wrap seam  : {verify.wrap_seam_error(result.image):.2f} / 255")
    report = verify.agreement(result.stats)
    print(report.report())

    if args.out is not None:
        size = io.write_png(args.out, result.image, compress_level=args.compress_level)
        print(f"wrote      : {args.out}  ({size / 2**20:.1f} MiB)")
    return 0 if report.passed else 1


def cmd_sequence(args: argparse.Namespace) -> int:
    candidates = [s for s in source_mod.scan(args.source) if s.usable] if args.source else []
    chosen = _select_stem(candidates, args.stem) if candidates else _resolve_source(args.source)
    try:
        rig = rig_for(chosen.camera_count)
    except UnknownRigError as exc:
        raise SystemExit(str(exc)) from None
    try:
        frames = pipeline.parse_frames(args.frames, chosen.frames)
        spec = encode.EncodeSpec.for_size(
            args.size,
            args.codec,
            args.bitrate,
            args.fps,
            preset=args.preset,
            deterministic=args.deterministic,
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
    except KeyboardInterrupt:
        print(
            "\ninterrupted; finished segments are kept. Re-run the same command to resume.",
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
    sequence.add_argument("--stats-every", type=int, default=100, help="geometry gate cadence")
    sequence.add_argument("--no-bar", action="store_true", help="plain log lines, no tqdm bar")
    sequence.add_argument("--no-resume", action="store_true", help="re-encode finished segments")
    sequence.add_argument("--keep-segments", action="store_true")
    sequence.set_defaults(func=cmd_sequence)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result
