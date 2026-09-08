"""Sequence job: decode -> warp -> encode, resumable, one MP4 out.

Three decisions shape this module (ROADMAP P2):

* **No PNG in the middle.** Frames go from the warp straight into ffmpeg's stdin as raw
  RGB. The 0.7-2.8 s/frame PNG encode that was the previous pipeline's ceiling simply
  does not exist here.
* **Segments, always.** An MP4 is one file, so "the output already exists" cannot express
  progress. The sequence is encoded as segments of whole GOPs, each its own ffmpeg run,
  and joined with `-c copy` at the end. A resumed run skips finished segments. Because
  every run -- interrupted or not -- is segmented identically, the joined result is the
  same file either way.
* **Decode runs ahead.** Pillow releases the GIL while decoding PNG, so a thread pool
  overlaps the next frame's 15 decodes (about 0.3 s wall) with the current frame's warp
  (about 1.7 s at 8K). The warp is the bottleneck; the decode is hidden behind it.

The geometry gate is sampled: every `stats_every` frames the overlap agreement is
computed and must pass, so a wrong rig cannot get through a long run unnoticed.
"""

from __future__ import annotations

import collections
import concurrent.futures
import dataclasses
import pathlib
import shutil
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import numpy.typing as npt

from vr_compose import encode, io, memory, verify
from vr_compose.rig import Rig
from vr_compose.source import SourceSet
from vr_compose.warp import DEFAULT_THREADS, WarpPlan

U8 = npt.NDArray[np.uint8]

__all__ = [
    "Cancelled",
    "GeometryGateFailed",
    "Progress",
    "SequenceJob",
    "Summary",
    "parse_frames",
    "run_sequence",
]

PREVIOUS_PIPELINE_SECONDS_PER_FRAME = 46.85
"""AGENTS.md §8: what the run replaces, for the progress line."""


class GeometryGateFailed(RuntimeError):
    """A sampled frame failed metric A. The rig is wrong; stop before wasting hours."""


class Cancelled(RuntimeError):
    """The run was stopped on purpose. Finished segments are kept; re-running resumes.

    A distinct type because the alternative -- inferring cancellation from whatever
    exception happens to escape -- got it wrong: a console Ctrl-C used to kill ffmpeg
    first (see :func:`vr_compose.encode.child_creation_flags`) and surfaced as
    "ffmpeg exited early", i.e. as a failure, with an empty message.
    """


class _CancelScope:
    """Turns Ctrl-C into a flag the frame loop checks, so no frame is cut in half.

    Only the main thread can carry a signal handler, and only the main thread's handler
    is ever installed -- a GUI (P6) or a test drives cancellation through the `cancel`
    event instead, which is the same mechanism without the signal.

    A second Ctrl-C restores the default handler, so an impatient user still gets the
    immediate KeyboardInterrupt they are asking for.
    """

    def __init__(self, event: threading.Event) -> None:
        self.event = event
        self._restore: object = None
        """What to put back, and the record that we installed anything at all."""

    def __enter__(self) -> _CancelScope:
        if threading.current_thread() is not threading.main_thread():
            return self
        try:
            previous = signal.signal(signal.SIGINT, self._handle)
        except ValueError:  # pragma: no cover -- not the main thread after all
            return self
        # `signal.signal` answers None when the handler in place was not set from Python
        # (an embedded interpreter, a C extension). There is no way to put that one back,
        # and leaving *ours* installed would silently swallow every later Ctrl-C, so fall
        # back to the handler CPython itself starts with.
        self._restore = previous if previous is not None else signal.default_int_handler
        return self

    def _handle(self, _signum: int, _frame: object) -> None:
        if self.event.is_set():
            self._put_back()  # a second Ctrl-C: let the next one interrupt immediately
        self.event.set()

    def _put_back(self) -> None:
        if self._restore is not None:
            signal.signal(signal.SIGINT, self._restore)  # type: ignore[arg-type]
            self._restore = None

    def __exit__(self, *_exc: object) -> None:
        self._put_back()


def parse_frames(text: str, available: Sequence[int]) -> list[int]:
    """`all`, `1656-2433`, `1656-`, `-1700`, or a comma list; restricted to `available`."""
    pool = set(available)
    if not pool:
        raise ValueError("the source set has no frames shared by every camera")
    text = text.strip().lower()
    if text in ("", "all"):
        return sorted(pool)
    chosen: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo_text, hi_text = part.split("-", 1)
            lo = int(lo_text) if lo_text else min(pool)
            hi = int(hi_text) if hi_text else max(pool)
            if hi < lo:
                raise ValueError(f"empty range {part!r}")
            chosen.update(f for f in pool if lo <= f <= hi)
        else:
            chosen.add(int(part))
    missing = sorted(chosen - pool)
    if missing:
        shown = ", ".join(map(str, missing[:6])) + (" ..." if len(missing) > 6 else "")
        raise ValueError(
            f"{len(missing)} requested frame(s) are not present in every camera: {shown}"
        )
    if not chosen:
        raise ValueError(f"no frames selected by {text!r}")
    return sorted(chosen)


@dataclasses.dataclass(frozen=True, slots=True)
class SequenceJob:
    source: SourceSet
    rig: Rig
    spec: encode.EncodeSpec
    frames: tuple[int, ...]
    output: pathlib.Path
    segment_gops: int = 5
    """Segment length in GOPs: 5 x 2 s = 10 s of video per ffmpeg run."""
    decode_workers: int = 4
    """Threads decoding the *next* frame's 15 tiles while the current one warps.

    Fewer than the obvious: the decode is already hidden behind the warp at 4 (measured
    wait 0.02 s), and 8 decode threads slow the warp itself from 2.0 s to 2.8 s by
    contending for the GIL between numpy calls. 4 decode + 8 warp threads measured best
    at 8K (2.48 s/frame end to end)."""
    warp_threads: int = DEFAULT_THREADS
    stats_every: int = 100
    """Run the geometry gate on frames 0, N, 2N, ... of the job."""
    resume: bool = True
    keep_segments: bool = False
    memory_soft_limit: int = memory.SOFT_LIMIT_BYTES
    """Advisory. Above this the run warns and carries on -- it never fails or stops."""

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("no frames to process")
        if self.segment_gops < 1:
            raise ValueError("segment_gops must be >= 1")
        if self.output.suffix.lower() != ".mp4":
            raise ValueError(f"output must be an .mp4 path, got {self.output}")

    @property
    def segment_frames(self) -> int:
        return self.segment_gops * self.spec.gop

    @property
    def segment_dir(self) -> pathlib.Path:
        return self.output.with_name(self.output.stem + ".segments")

    def segments(self) -> list[tuple[int, tuple[int, ...]]]:
        """(index, frames) per segment. Boundaries are whole GOPs from the job's first frame."""
        step = self.segment_frames
        return [
            (i, self.frames[start : start + step])
            for i, start in enumerate(range(0, len(self.frames), step))
        ]

    def segment_path(self, index: int, frames: tuple[int, ...]) -> pathlib.Path:
        return self.segment_dir / f"{self.output.stem}.{index:04d}.{frames[0]}-{frames[-1]}.mp4"

    def estimated_output_bytes(self) -> int:
        seconds = len(self.frames) / self.spec.fps
        video = self.spec.bitrate_kbps * 1000 / 8 * seconds
        # segments and the joined file coexist until cleanup, plus VBV overshoot headroom
        return int(video * 2 * 1.2)


@dataclasses.dataclass(frozen=True, slots=True)
class Progress:
    done: int
    total: int
    frame: int
    segment: int
    segments: int
    seconds_per_frame: float
    eta_seconds: float
    skipped: bool = False

    def line(self) -> str:
        speedup = (
            PREVIOUS_PIPELINE_SECONDS_PER_FRAME / self.seconds_per_frame
            if self.seconds_per_frame
            else 0
        )
        eta_min = self.eta_seconds / 60
        note = "  (segment already done)" if self.skipped else ""
        return (
            f"frame {self.frame}  {self.done}/{self.total}  "
            f"segment {self.segment + 1}/{self.segments}  "
            f"{self.seconds_per_frame:5.2f} s/frame ({speedup:4.1f}x vs 46.85)  "
            f"ETA {eta_min:5.1f} min{note}"
        )


@dataclasses.dataclass(slots=True)
class StageTimes:
    """Where the wall clock went, summed over encoded frames.

    `decode_wait` is time blocked on the prefetched tiles -- zero means the decode is fully
    hidden behind the warp, which is the design intent. Anything else here is on the
    critical path.
    """

    decode_wait: float = 0.0
    warp: float = 0.0
    gate: float = 0.0
    write: float = 0.0
    frames: int = 0

    def report(self) -> str:
        n = max(self.frames, 1)
        return (
            f"per frame: wait-for-decode {self.decode_wait / n:.2f} s, warp {self.warp / n:.2f} s, "
            f"gate {self.gate / n:.2f} s, write {self.write / n:.2f} s"
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Summary:
    output: pathlib.Path
    frames: int
    encoded: int
    skipped_segments: int
    seconds: float
    plan_seconds: float
    gate_reports: tuple[verify.Agreement, ...]
    stream: encode.StreamInfo
    problems: tuple[str, ...]
    stages: StageTimes
    peak: memory.PeakMemory = dataclasses.field(default_factory=memory.PeakMemory)
    warnings: tuple[str, ...] = ()
    """Advisory notes. Unlike `problems`, these do not make the run a failure."""

    @property
    def seconds_per_frame(self) -> float:
        return self.seconds / self.encoded if self.encoded else 0.0


def _check_disk(job: SequenceJob) -> None:
    target = job.output.parent
    target.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(target).free
    need = job.estimated_output_bytes()
    if free < need:
        raise RuntimeError(
            f"{target} has {free / 2**30:.1f} GiB free; this job needs about "
            f"{need / 2**30:.1f} GiB (segments + joined file + headroom). Refusing to start."
        )


def _segment_is_complete(tools: encode.Tools, path: pathlib.Path, expected_frames: int) -> bool:
    if not path.is_file():
        return False
    try:
        return encode.probe(tools, path).frames == expected_frames
    except (StopIteration, KeyError, ValueError):
        return False


def run_sequence(
    job: SequenceJob,
    *,
    progress: Callable[[Progress], None] | None = None,
    log: Callable[[str], None] | None = None,
    plan: WarpPlan | None = None,
    cancel: threading.Event | None = None,
) -> Summary:
    """Encode the whole job to `job.output`. Raises rather than producing a bad file.

    Setting `cancel` (or pressing Ctrl-C) stops the run at the next frame boundary with
    :class:`Cancelled`: the segment in progress is discarded, finished segments are kept,
    and re-running the same command resumes. Cancellation is checked between frames, so
    it takes up to one frame -- about 2 s at 8K -- to take effect.
    """
    say = log or (lambda _: None)
    tools = encode.find_tools()
    tools.require("libx264" if job.spec.codec == "h264" else "libx265")
    _check_disk(job)

    tile_size = job.source.tile_size
    if tile_size is None:
        raise ValueError("source tiles are not square or not uniform; cannot stitch")
    # The plan renders the *master*; ffmpeg resamples to the delivery size if they differ.
    master = (job.spec.master_width, job.spec.master_height)
    if plan is None:
        say("building warp plan ...")
        plan = WarpPlan.build(job.rig, *master, tile_size)
        say(f"plan ready in {plan.build_seconds:.1f} s ({plan.nbytes / 2**20:.0f} MiB)")
    elif (plan.width, plan.height, plan.tile_size) != (*master, tile_size):
        raise ValueError("the supplied warp plan does not match this job")

    indices = list(job.rig.unique_indices)
    segments = job.segments()
    total = len(job.frames)
    started = time.time()
    done = 0
    encoded = 0
    skipped = 0
    recent: collections.deque[float] = collections.deque(maxlen=20)
    reports: list[verify.Agreement] = []
    finished_paths: list[pathlib.Path] = []
    stages = StageTimes()
    encoder_commit = 0
    encoder_resident = 0

    def decode(frame: int) -> dict[int, U8]:
        return io.load_tiles(job.source, frame, indices, workers=job.decode_workers)

    stop = cancel if cancel is not None else threading.Event()
    with _CancelScope(stop), concurrent.futures.ThreadPoolExecutor(max_workers=1) as prefetch:
        for seg_index, seg_frames in segments:
            if stop.is_set():
                raise Cancelled(f"cancelled after {done} of {total} frame(s)")
            path = job.segment_path(seg_index, seg_frames)
            if job.resume and _segment_is_complete(tools, path, len(seg_frames)):
                skipped += 1
                done += len(seg_frames)
                finished_paths.append(path)
                if progress:
                    rate = float(np.mean(recent)) if recent else 0.0
                    progress(
                        Progress(done, total, seg_frames[-1], seg_index, len(segments), rate,
                                 rate * (total - done), skipped=True)
                    )  # fmt: skip
                continue

            writer = encode.SegmentWriter(tools, job.spec, path)
            pending = prefetch.submit(decode, seg_frames[0])
            try:
                for position, frame in enumerate(seg_frames):
                    if stop.is_set():
                        raise Cancelled(f"cancelled after {done} of {total} frame(s)")
                    tick = time.time()
                    tiles = pending.result()
                    if position + 1 < len(seg_frames):
                        pending = prefetch.submit(decode, seg_frames[position + 1])
                    waited = time.time()
                    want_stats = (done % job.stats_every) == 0
                    result = plan.apply(tiles, with_stats=want_stats, threads=job.warp_threads)
                    warped = time.time()
                    if want_stats:
                        report = verify.agreement(result.stats)
                        reports.append(report)
                        if not report.passed:
                            raise GeometryGateFailed(
                                f"frame {frame} failed the overlap-agreement gate:\n"
                                f"{report.report()}"
                            )
                    gated = time.time()
                    writer.write(result.image)
                    written = time.time()
                    stages.decode_wait += waited - tick
                    stages.warp += warped - waited
                    stages.gate += gated - warped
                    stages.write += written - gated
                    stages.frames += 1
                    done += 1
                    encoded += 1
                    recent.append(written - tick)
                    if progress:
                        rate = float(np.mean(recent))
                        progress(
                            Progress(done, total, frame, seg_index, len(segments), rate,
                                     rate * (total - done))
                        )  # fmt: skip
            except KeyboardInterrupt:
                # Reachable only if an interrupt slips past the scope's handler -- another
                # thread, or a second Ctrl-C that restored the default. Same outcome.
                writer.abort()
                raise Cancelled(f"cancelled after {done} of {total} frame(s)") from None
            except BaseException:
                writer.abort()
                raise
            finished_paths.append(writer.close())
            # segments run one at a time, so the largest is the job's encoder footprint
            encoder_commit = max(encoder_commit, writer.peak_commit)
            encoder_resident = max(encoder_resident, writer.peak_resident)
            say(f"segment {seg_index + 1}/{len(segments)} closed -- {stages.report()}")

    say(f"joining {len(finished_paths)} segment(s) ...")
    encode.concat(tools, finished_paths, job.output)
    stream = encode.probe(tools, job.output)
    problems = list(stream.conformance_problems(job.spec))
    if stream.frames != total:
        problems.append(f"joined file has {stream.frames} frames, expected {total}")
    if not job.keep_segments and not problems:
        shutil.rmtree(job.segment_dir, ignore_errors=True)
    peak = memory.PeakMemory(
        stitcher=memory.peak_commit(),
        encoder=encoder_commit,
        stitcher_resident=memory.peak_working_set(),
        encoder_resident=encoder_resident,
    )
    warnings: list[str] = []
    if peak.measured:
        say(peak.report())
        if peak.exceeds(job.memory_soft_limit):
            warnings.append(peak.warning(job.memory_soft_limit))
            say(f"WARNING: {warnings[-1]}")
    return Summary(
        output=job.output,
        frames=total,
        encoded=encoded,
        skipped_segments=skipped,
        seconds=time.time() - started,
        plan_seconds=plan.build_seconds,
        gate_reports=tuple(reports),
        stream=stream,
        problems=tuple(problems),
        stages=stages,
        peak=peak,
        warnings=tuple(warnings),
    )


def default_output_dir() -> pathlib.Path:
    """Where output lands when no path is given: beside the program.

    Frozen by PyInstaller that is the executable's directory; from source it is the
    project root. The GUI (P6) will offer a chooser; until then the user's decision is
    "next to where it runs".
    """
    if getattr(sys, "frozen", False):
        return pathlib.Path(sys.executable).parent
    return pathlib.Path(__file__).resolve().parents[2]


def default_output_name(source: SourceSet, spec: encode.EncodeSpec, frames: Sequence[int]) -> str:
    """`<stem>.<first>-<last>.<WxH>.<codec>.mp4` -- enough to tell renders apart."""
    return f"{source.stem}.{frames[0]}-{frames[-1]}.{spec.width}x{spec.height}.{spec.codec}.mp4"


def frames_in_segments(job: SequenceJob) -> Iterable[int]:
    """Convenience for callers that want the per-segment frame lists flattened."""
    for _, frames in job.segments():
        yield from frames
