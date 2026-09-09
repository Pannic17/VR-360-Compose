"""Command-line entry point, and the machine-readable surface the GUI drives.

The GUI in P6 must be a thin shell over exactly this code, so every operation lives in a
library module and this file only parses arguments and prints. Nothing here hardcodes a
source path: `--source` is optional and discovery fills it in (AGENTS.md §2, constraint 3).

"Thin shell" is taken literally: the GUI does not import the pipeline, it **runs this
CLI as a subprocess** (AGENTS.md §7 requires the job to be its own process, so a crash
cannot take the window with it and Qt never shares an interpreter with the worker pools).
Two options exist for that:

* `--progress-json` puts one JSON object per line on **stdout** and moves every human
  line to stderr, so the parent has a stream it can parse without scraping prose.
* `--cancel-on-stdin` lets the parent stop the job by writing a line -- or simply by
  closing the pipe, which also means "the GUI is gone, stop working". It sets the same
  `threading.Event` the pipeline already takes, so cancellation stays one mechanism with
  one set of guarantees (finished segments kept, no partial files, resumable).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import pathlib
import sys
import threading
import time
from collections.abc import Sequence

from tqdm import tqdm

from vr_compose import __version__, encode, io, pipeline, spherical, verify
from vr_compose import device as device_mod
from vr_compose import source as source_mod
from vr_compose.rig import Rig, UnknownRigError, rig_for
from vr_compose.stitch import (
    BIT_DEPTHS,
    DEFAULT_BAND_ROWS,
    DEFAULT_FEATHER_POWER,
    DEFAULT_SAMPLER,
    SAMPLERS,
    stitch_frame,
)
from vr_compose.warp import DEFAULT_THREADS, WarpPlan


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


@dataclasses.dataclass(slots=True)
class Reporter:
    """Where a run's progress and prose go: a terminal, or a parent process.

    In JSON mode stdout carries only NDJSON and prose goes to stderr, so a parent can
    parse one stream while still showing the other. In terminal mode this is a no-op
    wrapper and `print` behaves as it always did.
    """

    json: bool = False

    def event(self, kind: str, /, **fields: object) -> None:
        """`kind` is positional-only, so an event may also carry a `kind` field."""
        if self.json:
            print(json.dumps({"event": kind, **fields}, ensure_ascii=False), flush=True)

    def say(self, text: str) -> None:
        """A human line. Never stdout in JSON mode -- it would corrupt the stream."""
        print(text, file=sys.stderr if self.json else sys.stdout, flush=self.json)


def _watch_stdin_for_cancel(cancel: threading.Event) -> None:
    """Set `cancel` when the parent says so, or when it goes away.

    Any line means stop. So does EOF: if the parent closed the pipe it has either asked
    for this or died, and in both cases finishing the render serves nobody. Daemon
    thread, because a normal completion should not wait on a read that never returns.
    """

    def watch() -> None:
        with contextlib.suppress(Exception):
            for _line in sys.stdin:
                break
        cancel.set()

    threading.Thread(target=watch, name="cancel-watch", daemon=True).start()


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

    placement = device_mod.resolve_device(args.device)
    if placement.fell_back:
        print(f"WARNING    : {placement.warning}")
    elif placement.note:
        print(f"warp       : cpu ({placement.note})")
    if placement.device == "cuda":
        # The plan is the verified geometry evaluated once; the GPU only applies it. Same
        # bytes as `stitch_frame` -- tests/test_warp_gpu.py holds that line.
        from vr_compose import warp_gpu

        plan = WarpPlan.build(
            rig, width, width // 2, tile, band_rows=args.band_rows,
            sampler=args.sampler, feather_power=args.feather_power,
        )  # fmt: skip
        gpu = warp_gpu.GpuWarpPlan.from_plan(plan)
        print(
            f"warp       : cuda ({placement.detail}), plan built in {plan.build_seconds:.1f} s, "
            f"resident in {gpu.upload_seconds:.1f} s"
        )
        result = gpu.apply(tiles, with_stats=True, bit_depth=args.bit_depth)
    else:
        result = stitch_frame(
            tiles,
            rig,
            width,
            band_rows=args.band_rows,
            sampler=args.sampler,
            feather_power=args.feather_power,
            bit_depth=args.bit_depth,
        )
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
        size = io.write_png_atomically(args.out, result.image, compress_level=args.compress_level)
        print(f"wrote      : {args.out}  ({size / 2**20:.1f} MiB)")
    return 0 if report.passed else 1


def _describe_device(device: str, detail: str) -> str:
    return f"{device} ({detail})" if detail else device


DEVICE_HELP = (
    "where the warp runs. 'cpu' (default) is the byte-exact reference; 'cuda' gives the "
    "same bytes from an NVIDIA GPU with at least 12 GB (needs `pip install "
    "vr-compose[gpu]`) and warns and uses the CPU without one; 'auto' (what the GUI "
    "sends) takes such a GPU when present and the CPU otherwise, quietly"
)

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
    "no_spherical",
)
"""`sequence` options that only mean something when an encoder is involved."""

MASTER_ONLY = ("compress_level", "write_workers", "bit_depth")
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
    args: argparse.Namespace,
    chosen: source_mod.SourceSet,
    rig: Rig,
    frames: list[int],
    stamp: str,
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
    default_name = pipeline.default_master_dir_name(chosen, stamp)
    try:
        job = pipeline.MasterJob(
            source=chosen,
            rig=rig,
            frames=tuple(frames),
            directory=args.out
            or pipeline.unique_path(pipeline.default_output_dir() / default_name),
            width=width,
            compress_level=args.compress_level,
            decode_workers=args.decode_workers,
            warp_threads=args.warp_threads,
            sampler=args.sampler,
            feather_power=args.feather_power,
            bit_depth=args.bit_depth,
            write_workers=args.write_workers,
            stats_every=args.stats_every,
            resume=not args.no_resume,
            device=args.device,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    pending = job.pending_frames()
    report = Reporter(json=args.progress_json)
    report.say(f"source     : {chosen.root}  stem {chosen.stem!r}")
    report.say(
        f"rig        : {rig.name}, reading {len(rig.unique_indices)} of {rig.file_count} files"
    )
    report.say(f"frames     : {frames[0]}..{frames[-1]} ({len(frames)}), {len(pending)} to render")
    report.say(
        f"master     : {width}x{width // 2} {job.bit_depth}-bit PNG at the input's own "
        f"density, lossless (zlib {job.compress_level}), sampler {job.sampler}"
    )
    report.say(f"output     : {job.directory}")
    report.event(
        "start",
        mode="frames",
        total=len(frames),
        pending=len(pending),
        first=frames[0],
        last=frames[-1],
        output=str(job.directory),
        describe=f"{width}x{width // 2} {job.bit_depth}-bit PNG, lossless",
        sampler=job.sampler,
        baseline_seconds_per_frame=pipeline.PREVIOUS_PIPELINE_SECONDS_PER_FRAME,
    )
    cancel = threading.Event()
    if args.cancel_on_stdin:
        _watch_stdin_for_cancel(cancel)

    bar = tqdm(
        total=len(frames),
        unit="frame",
        desc="mastering",
        dynamic_ncols=True,
        disable=args.no_bar,
        mininterval=1.0,
    )
    bar.update(len(frames) - len(pending))

    def on_progress(progress: pipeline.Progress) -> None:
        bar.n = progress.done
        bar.set_postfix_str(f"{progress.seconds_per_frame:.2f} s/frame", refresh=False)
        bar.refresh()
        speedup = (
            pipeline.PREVIOUS_PIPELINE_SECONDS_PER_FRAME / progress.seconds_per_frame
            if progress.seconds_per_frame
            else 0.0
        )
        report.event(
            "progress",
            done=progress.done,
            total=progress.total,
            frame=progress.frame,
            segment=1,
            segments=1,
            seconds_per_frame=round(progress.seconds_per_frame, 3),
            eta_seconds=round(progress.eta_seconds, 1),
            speedup=round(speedup, 2),
            skipped=progress.skipped,
        )

    def on_log(message: str) -> None:
        if args.progress_json:
            report.event("log", message=message)
        else:
            tqdm.write(f"           {message}")

    try:
        summary = pipeline.run_master(job, progress=on_progress, log=on_log, cancel=cancel)
    except pipeline.GeometryGateFailed as exc:
        report.event("error", kind="geometry", message=str(exc))
        raise SystemExit(
            f"geometry gate FAILED -- stopping before wasting the run:\n{exc}"
        ) from None
    except (pipeline.Cancelled, KeyboardInterrupt) as exc:
        detail = f" ({exc})" if isinstance(exc, pipeline.Cancelled) else ""
        report.event("cancelled", message=str(exc), output=str(job.directory))
        print(
            f"\ninterrupted{detail}; finished masters are kept. Re-run the same command to resume.",
            file=sys.stderr,
        )
        return 130
    except RuntimeError as exc:
        report.event("error", kind="failure", message=str(exc))
        raise SystemExit(str(exc)) from None
    finally:
        bar.close()

    report.say(f"\ndone       : {summary.written} master(s) -> {summary.directory}")
    if summary.written:
        report.say(
            f"throughput : {summary.seconds_per_frame:.2f} s/frame, "
            f"{summary.bytes_per_frame / 2**20:.1f} MiB/frame "
            f"({summary.bytes_written / 2**30:.2f} GiB written)"
        )
        report.say(f"stages     : {summary.stages.report()}")
    report.say(f"warp       : {_describe_device(summary.device, summary.device_detail)}")
    if summary.skipped:
        report.say(f"resumed    : {summary.skipped} master(s) were already present")
    if summary.gate_reports:
        worst = max(gate.median for gate in summary.gate_reports)
        gates = len(summary.gate_reports)
        report.say(f"geometry   : {gates} sampled frame(s) PASS, worst median {worst:.2f}")
    if summary.peak.measured:
        report.say(f"memory     : {summary.peak.report()}")
    for warning in summary.warnings:
        report.say(f"WARNING    : {warning}")
    report.event(
        "done",
        ok=True,
        output=str(summary.directory),
        frames=summary.frames,
        written=summary.written,
        skipped=summary.skipped,
        seconds_per_frame=round(summary.seconds_per_frame, 3),
        size_bytes=summary.bytes_written,
        device=summary.device,
        problems=[],
        warnings=list(summary.warnings),
    )
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
    # One stamp for the whole run: it names the output *and*, for the MP4 path, the
    # segment directory derived from it, so a second reading could split a job in two.
    stamp = pipeline.run_stamp()
    if args.out_format == "exr":
        raise SystemExit(
            "EXR masters are not implemented: the source is 8-bit PNG, so an EXR would "
            "carry no more information than the PNG master does. The interface is "
            "reserved for a 16-bit source, which needs a change upstream that is not "
            "available (UPSTREAM.md, item 3). Use --out-format png."
        )
    if args.out_format == "png":
        return cmd_master(args, chosen, rig, frames, stamp)
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
            or pipeline.unique_path(
                pipeline.default_output_dir() / pipeline.default_output_name(chosen, stamp)
            ),
            segment_gops=args.segment_gops,
            decode_workers=args.decode_workers,
            warp_threads=args.warp_threads,
            sampler=args.sampler,
            feather_power=args.feather_power,
            stats_every=args.stats_every,
            resume=not args.no_resume,
            keep_segments=args.keep_segments,
            spherical=None if args.no_spherical else spherical.Spherical(),
            device=args.device,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    report = Reporter(json=args.progress_json)
    report.say(f"source     : {chosen.root}  stem {chosen.stem!r}")
    report.say(
        f"rig        : {rig.name}, reading {len(rig.unique_indices)} of {rig.file_count} files"
    )
    report.say(
        f"frames     : {frames[0]}..{frames[-1]} ({len(frames)}), "
        f"{len(job.segments())} segment(s) of up to {job.segment_frames}"
    )
    report.say(f"sampler    : {args.sampler}")
    report.say(f"encode     : {spec.describe()}")
    report.say(f"output     : {job.output}")
    report.event(
        "start",
        mode="video",
        total=len(frames),
        first=frames[0],
        last=frames[-1],
        output=str(job.output),
        describe=spec.describe(),
        sampler=args.sampler,
        baseline_seconds_per_frame=pipeline.PREVIOUS_PIPELINE_SECONDS_PER_FRAME,
    )
    cancel = threading.Event()
    if args.cancel_on_stdin:
        _watch_stdin_for_cancel(cancel)

    bar = tqdm(
        total=len(frames),
        unit="frame",
        desc="stitching",
        dynamic_ncols=True,
        disable=args.no_bar,
        mininterval=1.0,
    )

    def on_progress(progress: pipeline.Progress) -> None:
        bar.n = progress.done
        speedup = (
            pipeline.PREVIOUS_PIPELINE_SECONDS_PER_FRAME / progress.seconds_per_frame
            if progress.seconds_per_frame
            else 0.0
        )
        bar.set_postfix_str(
            f"seg {progress.segment + 1}/{progress.segments}  "
            f"{progress.seconds_per_frame:.2f} s/frame  {speedup:.1f}x vs 46.85",
            refresh=False,
        )
        bar.refresh()
        report.event(
            "progress",
            done=progress.done,
            total=progress.total,
            frame=progress.frame,
            segment=progress.segment + 1,
            segments=progress.segments,
            seconds_per_frame=round(progress.seconds_per_frame, 3),
            eta_seconds=round(progress.eta_seconds, 1),
            speedup=round(speedup, 2),
            skipped=progress.skipped,
        )

    def on_log(message: str) -> None:
        # One channel each, or a parent showing both would print every line twice: in
        # JSON mode these go out as events, and the header lines above stay on stderr.
        if args.progress_json:
            report.event("log", message=message)
        else:
            # tqdm.write keeps the bar intact; a bare print would smear it
            tqdm.write(f"           {message}")

    try:
        summary = pipeline.run_sequence(job, progress=on_progress, log=on_log, cancel=cancel)
    except pipeline.GeometryGateFailed as exc:
        report.event("error", kind="geometry", message=str(exc))
        raise SystemExit(
            f"geometry gate FAILED -- stopping before wasting the run:\n{exc}"
        ) from None
    # Cancelled is a RuntimeError, so it has to be caught before the failure branch below.
    except (pipeline.Cancelled, KeyboardInterrupt) as exc:
        detail = f" ({exc})" if isinstance(exc, pipeline.Cancelled) else ""
        report.event("cancelled", message=str(exc), output=str(job.output))
        print(
            f"\ninterrupted{detail}; finished segments are kept. "
            "Re-run the same command to resume.",
            file=sys.stderr,
        )
        return 130
    except (encode.FfmpegNotFound, RuntimeError) as exc:
        report.event("error", kind="failure", message=str(exc))
        raise SystemExit(str(exc)) from None
    finally:
        bar.close()

    size_mib = summary.output.stat().st_size / 2**20
    report.say(f"\ndone       : {summary.frames} frames -> {summary.output} ({size_mib:.0f} MiB)")
    if summary.encoded:
        speedup = pipeline.PREVIOUS_PIPELINE_SECONDS_PER_FRAME / summary.seconds_per_frame
        report.say(
            f"throughput : {summary.seconds_per_frame:.2f} s/frame over {summary.encoded} "
            f"encoded frame(s) ({speedup:.1f}x vs the previous pipeline's 46.85)"
        )
        report.say(f"stages     : {summary.stages.report()}")
    report.say(f"warp       : {_describe_device(summary.device, summary.device_detail)}")
    if summary.peak.measured:
        report.say(f"memory     : {summary.peak.report()}")
    if summary.skipped_segments:
        report.say(f"resumed    : {summary.skipped_segments} segment(s) were already complete")
    gates = summary.gate_reports
    if gates:
        worst = max(gate.median for gate in gates)
        report.say(f"geometry   : {len(gates)} sampled frame(s) PASS, worst median {worst:.2f}")
    stream = summary.stream
    report.say(
        f"stream     : {stream.codec} {stream.profile} L{stream.level} {stream.tag} "
        f"{stream.width}x{stream.height} {stream.pix_fmt} B={stream.b_frames} "
        f"I-gap={list(stream.i_intervals)} {stream.bitrate_mbps:.1f} Mbps"
    )
    report.say(f"spherical  : {stream.projection or 'absent (--no-spherical)'}")
    for warning in summary.warnings:
        report.say(f"WARNING    : {warning}")
    report.event(
        "done",
        ok=not summary.problems,
        output=str(summary.output),
        frames=summary.frames,
        encoded=summary.encoded,
        seconds_per_frame=round(summary.seconds_per_frame, 3),
        size_bytes=summary.output.stat().st_size,
        device=summary.device,
        problems=list(summary.problems),
        warnings=list(summary.warnings),
    )
    if summary.problems:
        report.say("CONFORMANCE PROBLEMS:")
        for problem in summary.problems:
            report.say(f"  - {problem}")
        return 1
    report.say("conformance: OK")
    return 0


def cmd_metadata(args: argparse.Namespace) -> int:
    """Report -- or add -- the Spherical Video V2 metadata on an MP4 that already exists.

    `sequence` writes it as part of the job, so this is for files rendered before the
    metadata existed, and for checking a delivery without re-rendering it.
    """
    path: pathlib.Path = args.file
    if not path.is_file():
        raise SystemExit(f"no such file: {path}")
    try:
        present = spherical.read(path)
        if args.write and present is None:
            added = spherical.inject(path)
            present = spherical.read(path)
            print(f"wrote {added} bytes of spherical metadata into {path.name}")
        elif args.write:
            print(f"{path.name} already carries spherical metadata; left untouched")
    except spherical.SphericalError as exc:
        raise SystemExit(str(exc)) from None
    print(f"metadata   : {present.describe() if present else 'absent'}")
    try:
        tools = encode.find_tools()
    except encode.FfmpegNotFound:
        return 0 if present else 1
    # ffprobe is an independent reader of the same boxes; if the two disagree, believe it.
    stream = encode.probe(tools, path)
    print(f"ffprobe    : projection {stream.projection or 'absent'}")
    if bool(present) != bool(stream.projection):
        raise SystemExit("our parser and ffprobe disagree; the boxes are malformed")
    return 0 if present else 1


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

    metadata = sub.add_parser("metadata", help="report or add the 360 metadata on an MP4")
    metadata.add_argument("file", type=pathlib.Path)
    metadata.add_argument(
        "--write", action="store_true", help="add the metadata if the file has none"
    )
    metadata.set_defaults(func=cmd_metadata)

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
        "--device", choices=list(device_mod.DEVICES), default=device_mod.DEFAULT_DEVICE,
        help=DEVICE_HELP,
    )  # fmt: skip
    frame.add_argument(
        "--bit-depth", type=int, default=8, choices=list(BIT_DEPTHS), help="output PNG depth"
    )
    frame.add_argument(
        "--feather-power", type=float, default=DEFAULT_FEATHER_POWER, help="blend exponent"
    )
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
        help="output .mp4, or the directory for --out-format png. Default: "
        "<stem>_MMDD_HHMM beside the program -- a new name every run, so an "
        "auto-named run does not resume; pass this explicitly to continue a previous one",
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
        default=None,
        help="threads decoding the next frame. Default: 4 on the CPU (more slows the warp, "
        "GIL contention), 8 on the GPU (there the decode is the bottleneck)",
    )
    sequence.add_argument("--warp-threads", type=int, default=DEFAULT_THREADS)
    sequence.add_argument(
        "--device", choices=list(device_mod.DEVICES), default=device_mod.DEFAULT_DEVICE,
        help=DEVICE_HELP,
    )  # fmt: skip
    sequence.add_argument(
        "--feather-power",
        type=float,
        default=DEFAULT_FEATHER_POWER,
        help="exponent on a tile's distance-to-edge when blending; higher prefers the "
        "tile that sees a direction most centrally. Measured optimum is 2-4",
    )
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
    sequence.add_argument(
        "--progress-json",
        action="store_true",
        help="one JSON object per line on stdout (start / progress / log / done / "
        "cancelled / error), with every human line moved to stderr. This is the surface "
        "the GUI reads; it is stable enough to script against",
    )
    sequence.add_argument(
        "--cancel-on-stdin",
        action="store_true",
        help="stop cleanly when a line arrives on stdin, or when stdin reaches EOF -- "
        "which also covers 'the parent process is gone'. Finished work is kept and the "
        "job stays resumable, exactly as with Ctrl-C",
    )
    sequence.add_argument(
        "--no-spherical",
        action="store_true",
        help="leave out the Spherical Video V2 metadata; the file then plays as a flat "
        "2:1 video instead of a 360 panorama (--out-format mp4 only)",
    )
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
        "--bit-depth",
        type=int,
        default=8,
        choices=list(BIT_DEPTHS),
        help="PNG master depth; 16 keeps the resample's sub-level precision, which 8-bit "
        "quantisation throws away (--out-format png only)",
    )
    sequence.add_argument(
        "--write-workers",
        type=int,
        default=None,
        help="PNG encodes in flight (--out-format png only). Default: 1 on the CPU (a "
        "second worker slows the warp), 4 on the GPU (there the PNG encode is the "
        "bottleneck: 8-bit 1.26 -> 0.51 s/frame)",
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
