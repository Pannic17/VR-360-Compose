"""Sequence jobs: decode -> warp -> sink, resumable. One MP4 out, or PNG masters.

Two sinks share one core. :func:`run_sequence` is the delivery path -- straight into
ffmpeg, no intermediate file. :func:`run_master` is the archival one, a lossless PNG per
frame, which P4 needs to compare quality against. Both get their frames from
:func:`_prefetch_tiles` and :func:`_stitch_one`, so the geometry, the decode overlap and
the sampled gate are the same code in both.

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
import contextlib
import dataclasses
import datetime
import itertools
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
from vr_compose import spherical as spherical_mod
from vr_compose.rig import Rig
from vr_compose.source import SourceSet
from vr_compose.stitch import BIT_DEPTHS, DEFAULT_FEATHER_POWER, DEFAULT_SAMPLER
from vr_compose.warp import DEFAULT_THREADS, WarpPlan

U8 = npt.NDArray[np.uint8]

__all__ = [
    "Cancelled",
    "GeometryGateFailed",
    "MasterJob",
    "MasterSummary",
    "Progress",
    "SequenceJob",
    "Summary",
    "contiguous_tail",
    "parse_frames",
    "run_master",
    "run_sequence",
    "run_stamp",
    "unique_path",
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


def contiguous_tail(frames: Sequence[int]) -> tuple[int, int]:
    """`(first, last)` of the final run of consecutive frames.

    The reference set is `0000` plus `1656..2433`: frames 1 to 1655 were consumed and
    deleted by the previous pipeline, and `0000` is a leftover reference frame that is
    not part of the sequence (AGENTS.md section 4). Rendering "everything" therefore
    produces a video that jumps after its first frame, which is why the docs tell people
    not to write `--frames all`.

    Telling them is weaker than defaulting well, so this is what the GUI prefills. The
    CLI's `all` still means all -- changing what an explicit word means would be worse
    than the trap it avoids -- but both front ends can now reach the same answer.
    """
    if not frames:
        raise ValueError("no frames to choose from")
    index = len(frames) - 1
    while index > 0 and frames[index - 1] == frames[index] - 1:
        index -= 1
    return frames[index], frames[-1]


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
    sampler: str = DEFAULT_SAMPLER
    """How a tile is read at a fractional position; see :data:`vr_compose.stitch.SAMPLERS`."""
    feather_power: float = DEFAULT_FEATHER_POWER
    stats_every: int = 100
    """Run the geometry gate on frames 0, N, 2N, ... of the job."""
    resume: bool = True
    keep_segments: bool = False
    spherical: spherical_mod.Spherical | None = dataclasses.field(
        default_factory=spherical_mod.Spherical
    )
    """Spherical Video V2 metadata written into the joined file, or `None` to leave it out.

    Without it a player has no way to know the file is a projection rather than a very
    wide flat video, so it is on by default and the conformance check demands it. The
    metadata goes on the *joined* file only: `-c copy` does not carry it across a concat,
    so annotating segments would achieve nothing."""
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


def _prefetch_tiles(
    pool: concurrent.futures.ThreadPoolExecutor,
    load: Callable[[int], dict[int, U8]],
    frames: Sequence[int],
) -> Iterable[tuple[int, dict[int, U8], float]]:
    """Yield `(frame, tiles, seconds blocked)`, decoding the next frame while you work.

    One frame of lookahead, which is all that is useful: at 8K the 15 decodes take about
    0.3 s against a 1.9 s warp, so the measured wait is 0.01 s. Shared by the MP4 and the
    master paths so both get the overlap and neither reimplements it.
    """
    if not frames:
        return
    pending = pool.submit(load, frames[0])
    for position, frame in enumerate(frames):
        tick = time.time()
        tiles = pending.result()
        if position + 1 < len(frames):
            pending = pool.submit(load, frames[position + 1])
        yield frame, tiles, time.time() - tick


def _stitch_one(
    plan: WarpPlan, tiles: dict[int, U8], *, threads: int, gate: bool, bit_depth: int = 8
) -> tuple[io.Panorama, verify.Agreement | None, float, float]:
    """One frame: `(image, gate report or None, warp seconds, gate seconds)`."""
    started = time.time()
    result = plan.apply(tiles, with_stats=gate, threads=threads, bit_depth=bit_depth)
    warped = time.time()
    report = verify.agreement(result.stats) if gate else None
    return result.image, report, warped - started, time.time() - warped


def _delivery_frame(image: io.Panorama) -> U8:
    """Narrow a panorama to the 8-bit frame ffmpeg is fed.

    The MP4 path is 8-bit by construction -- `bit_depth` is a master-only option, and the
    delivery spec pins 8-bit 4:2:0 anyway -- so this is a guard rather than a conversion.
    Casting silently would turn a future mis-wiring into a 256x-too-dark video.
    """
    if image.dtype != np.uint8:
        raise ValueError(f"the delivery path needs 8-bit frames, got {image.dtype}")
    return np.asarray(image, dtype=np.uint8)


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
        plan = WarpPlan.build(
            job.rig, *master, tile_size, sampler=job.sampler, feather_power=job.feather_power
        )
        say(
            f"plan ready in {plan.build_seconds:.1f} s ({plan.nbytes / 2**20:.0f} MiB, "
            f"{plan.sampler})"
        )
    elif (plan.width, plan.height, plan.tile_size, plan.sampler, plan.feather_power) != (
        *master,
        tile_size,
        job.sampler,
        job.feather_power,
    ):
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
            try:
                for frame, tiles, waited in _prefetch_tiles(prefetch, decode, seg_frames):
                    if stop.is_set():
                        raise Cancelled(f"cancelled after {done} of {total} frame(s)")
                    image, report, warp_seconds, gate_seconds = _stitch_one(
                        plan,
                        tiles,
                        threads=job.warp_threads,
                        gate=(done % job.stats_every) == 0,
                    )
                    if report is not None:
                        reports.append(report)
                        if not report.passed:
                            raise GeometryGateFailed(
                                f"frame {frame} failed the overlap-agreement gate:\n"
                                f"{report.report()}"
                            )
                    write_started = time.time()
                    writer.write(_delivery_frame(image))
                    written = time.time()
                    stages.decode_wait += waited
                    stages.warp += warp_seconds
                    stages.gate += gate_seconds
                    stages.write += written - write_started
                    stages.frames += 1
                    done += 1
                    encoded += 1
                    recent.append(waited + warp_seconds + gate_seconds + written - write_started)
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
    metadata_failure: str | None = None
    if job.spherical is not None:
        try:
            added = spherical_mod.inject(job.output, job.spherical)
            say(f"spherical metadata: {job.spherical.describe()} (+{added} bytes)")
        except spherical_mod.SphericalError as exc:
            # The frames are already encoded and joined, and the file is valid -- it just
            # has no metadata. Failing the run here would throw away half an hour of
            # rendering over 104 bytes that `vr-compose metadata --write` can add later,
            # so this is reported as a conformance problem instead: non-zero exit, file
            # kept, and the message says what to do about it.
            metadata_failure = f"spherical metadata not written ({exc}); try `metadata --write`"
            say(f"WARNING: {metadata_failure}")
    stream = encode.probe(tools, job.output)
    problems = list(stream.conformance_problems(job.spec, spherical=job.spherical is not None))
    if metadata_failure:
        problems.append(metadata_failure)
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


def run_stamp(when: datetime.datetime | None = None) -> str:
    """`MMDD_HHMM` in local time, for the auto-generated output names (P5).

    No colons: they are illegal in Windows filenames. Local time rather than UTC because
    the name is for the person who started the render.

    **The user asked for exactly this width** (2026-09-08), and it costs two properties
    the longer `YYYYMMDD_HHMMSS` had, both worth naming:

    * **It is only sortable within a year.** `0102` sorts before `1231` regardless of
      which year each one is, so a folder spanning New Year sorts wrong. Accepted: the
      name is a label for the person who started the render, not an archive key.
    * **It is no longer unique.** Two runs in the same minute produce the same name,
      which for an auto-generated path used to be impossible. That one is *not* accepted,
      because P5 promises an auto-generated name is a fresh output every run and the MP4
      path derives its segment directory from it -- a collision would silently resume
      somebody else's segments, possibly encoded with different settings. `unique_path`
      below restores the promise.

    **Call this once per run and pass the result around.** A second call can land in the
    next minute, and for the MP4 path the name also determines the segment directory, so
    two stamps in one job would scatter the segments across two places.
    """
    return (when or datetime.datetime.now()).strftime("%m%d_%H%M")


def unique_path(path: pathlib.Path) -> pathlib.Path:
    """`path` if it is free, else the same name with `_2`, `_3`, ... before the suffix.

    Only ever applied to an **auto-generated** path. An explicit `--out` is left exactly
    as the user typed it, because reusing a path is how resume is asked for (P3) -- the
    whole difference between the two is that one resumes and the other does not.

    Minute-resolution stamps made the collision reachable: cancel a run, change a
    setting, start again inside the same minute, and without this the second run would
    adopt the first run's finished segments. Suffixing is the quiet fix; the alternative
    -- refusing, or resuming with mismatched settings -- is worse in both directions.
    """
    if not path.exists():
        return path
    for n in itertools.count(2):
        candidate = path.with_name(f"{path.stem}_{n}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise AssertionError("unreachable: itertools.count is infinite")


def default_output_name(source: SourceSet, stamp: str) -> str:
    """`<stem>_MMDD_HHMM.mp4` -- the naming the user asked for in P5.

    It replaced `<stem>.<first>-<last>.<WxH>.<codec>.mp4`, which said more about the
    render but could not tell two runs of the same command apart. A timestamp can, and
    that is the trade: **an auto-generated name is a new file every run, so it does not
    resume.** Pass `--out` explicitly to resume a previous one; the run prints the path
    it chose for exactly that purpose.
    """
    return f"{source.stem}_{stamp}.mp4"


def default_master_dir_name(source: SourceSet, stamp: str) -> str:
    """`<stem>_MMDD_HHMM` -- the directory a frame-mode run fills (P5)."""
    return f"{source.stem}_{stamp}"


def frames_in_segments(job: SequenceJob) -> Iterable[int]:
    """Convenience for callers that want the per-segment frame lists flattened."""
    for _, frames in job.segments():
        yield from frames


# --- master sink ----------------------------------------------------------------------
#
# A lossless PNG sequence at the source's native density: the input to P4's quality work
# and to archiving, not a delivery format. It shares this module's decode prefetch, warp
# and geometry gate with the MP4 path, and differs in three ways that matter:
#
# * **No segments.** One file per frame is already the unit of progress, so resuming is
#   "does this frame's file exist" -- checked before the decode is even queued, so a
#   resumed run does no work it will throw away.
# * **The write is not free.** A 7680x3840 PNG costs 0.68 s at compress_level 1 and
#   2.77 s at 6 (AGENTS.md section 8), against a 1.9 s warp. Level 1 by default, and the
#   writes go to a small thread pool -- Pillow releases the GIL inside the encoder -- so
#   they overlap the next frame's warp instead of adding to it.
# * **Disk is the constraint, not bitrate.** 39.6 MiB a frame measured at 8 bits, so the
#   778-frame job is 30 GiB -- fifteen times the MP4 -- and about 101 GiB at 16 bits. The
#   precheck refuses up front rather than dying half way through.

MASTER_BYTES_PER_PIXEL = {8: 1.6, 16: 5.0}
"""Conservative estimate per bit depth, for the disk precheck only.

Real 8K masters of this content measure **39.6 MiB a frame** at 8 bits (1.41
bytes/pixel) and **133.4 MiB at 16** (4.53) -- lossless PNG on high-frequency lava
geometry gets less than a 2:1 saving over raw RGB, and the extra depth is the resample's
sub-level detail, which is noise-like and barely compresses. The "Up" filter in
:func:`vr_compose.io.write_png16` takes the 16-bit figure down from 153.9 MiB, a 13%
saving rather than the halving one might hope for, for the same reason.

Each figure carries headroom over the measurement so a busier frame cannot make the
precheck under-reserve. A 778-frame 8K master set is about 30 GiB at 8 bits and
**101 GiB at 16** -- which is the number worth knowing before starting one."""


@dataclasses.dataclass(frozen=True, slots=True)
class MasterJob:
    """Render `frames` as a lossless PNG sequence into `directory`."""

    source: SourceSet
    rig: Rig
    frames: tuple[int, ...]
    directory: pathlib.Path
    width: int
    """Master width, 2:1, and it **must** be the source's native density.

    The user's P5 constraint: frame mode outputs at the input's own size and applies no
    compression. Both halves are structural here rather than conventional --
    :meth:`__post_init__` refuses any other width, and this sink has no codec, no
    bitrate and no delivery-size option to apply even if one were passed. The only
    resizing in the project happens on the *delivery* path, inside ffmpeg
    (:attr:`vr_compose.encode.EncodeSpec.master`).

    "Native density" means `Rig.native_width(tile_size)`, which for a 90-degree rig is
    four times the tile: 1920-pixel tiles give a 7680x3840 master. See AGENTS.md
    section 3 -- that width is where the panorama samples the render neither more nor
    less finely than it was drawn."""
    compress_level: int = 1
    """zlib level for the PNG. **Lossless at every setting**, so it is not "compression"
    in the sense that costs quality: level 0 stores the samples verbatim and level 9
    packs them hardest, and every level decodes to the same pixels -- `tests/
    test_master.py` asserts that against the stitcher's output.

    1 by default: it costs 0.68 s a frame against 6's 2.77 s for about 10% more bytes
    (AGENTS.md section 8), and a master is an intermediate, so time is worth more than
    the bytes. Pass 0 for literally uncompressed files, at roughly 2.3x the disk."""
    decode_workers: int = 4
    warp_threads: int = DEFAULT_THREADS
    sampler: str = DEFAULT_SAMPLER
    feather_power: float = DEFAULT_FEATHER_POWER
    bit_depth: int = 8
    """8 for a delivery-comparable master, 16 to keep the resample's sub-level precision.

    The source is 8-bit, so 16 bits add nothing *from the render*; what they keep is what
    the interpolation and the blend produced between levels, which 8-bit quantisation
    discards. Worth it for a master something later resamples again -- and it is what
    makes an eventual EXR source a drop-in (ROADMAP P4 item 6)."""
    write_workers: int = 1
    """PNG encodes in flight.

    One is fastest, measured: at 8K, 1 writer gives 2.40 s/frame (warp 2.06, blocked on
    the write 0.12) against 2 writers' 2.52 (warp 2.28, blocked 0.02). The second worker
    hides the write almost completely and pays for it by slowing the warp -- the same
    trade AGENTS.md section 8 records for the decode pool, where the *total* thread count
    mattered more than the split. Raise it only if the disk, not the CPU, is the limit.

    Also bounds memory: each queued image is 88 MB at 8K."""
    stats_every: int = 100
    resume: bool = True
    memory_soft_limit: int = memory.SOFT_LIMIT_BYTES

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("no frames to process")
        if self.width < 2 or self.width % 2:
            raise ValueError(f"master width must be even and positive, got {self.width}")
        tile = self.source.tile_size
        if tile is not None and self.width != self.rig.native_width(tile):
            raise ValueError(
                f"a master is written at the input's own density: {tile}px tiles give "
                f"{self.rig.native_width(tile)} wide, not {self.width}. Frame mode has no "
                "delivery size -- use `frame --width` for a one-off at another size."
            )
        if not 0 <= self.compress_level <= 9:
            raise ValueError(f"compress_level must be 0..9, got {self.compress_level}")
        if self.bit_depth not in BIT_DEPTHS:
            raise ValueError(f"bit_depth must be one of {BIT_DEPTHS}, got {self.bit_depth}")
        if self.write_workers < 1:
            raise ValueError("write_workers must be >= 1")

    @property
    def height(self) -> int:
        return self.width // 2

    def frame_path(self, frame: int) -> pathlib.Path:
        """`<stem>_S_<frame>.png`, the naming the user asked for in P5.

        The frame number keeps the source's own zero padding, so a directory listing
        sorts the way the source does -- unpadded, `999` would sort after `1000`.
        """
        return self.directory / f"{self.source.stem}_S_{frame:0{self.source.frame_digits}d}.png"

    def pending_frames(self) -> tuple[int, ...]:
        """Frames still to render. Consulted *before* decoding, not after."""
        if not self.resume:
            return self.frames
        return tuple(f for f in self.frames if not self.frame_path(f).is_file())

    def estimated_bytes(self, frames: int | None = None) -> int:
        count = len(self.frames) if frames is None else frames
        rate = MASTER_BYTES_PER_PIXEL[self.bit_depth]
        return int(count * self.width * self.height * rate)


@dataclasses.dataclass(frozen=True, slots=True)
class MasterSummary:
    directory: pathlib.Path
    frames: int
    written: int
    skipped: int
    seconds: float
    plan_seconds: float
    bytes_written: int
    gate_reports: tuple[verify.Agreement, ...]
    stages: StageTimes
    peak: memory.PeakMemory = dataclasses.field(default_factory=memory.PeakMemory)
    warnings: tuple[str, ...] = ()

    @property
    def seconds_per_frame(self) -> float:
        return self.seconds / self.written if self.written else 0.0

    @property
    def bytes_per_frame(self) -> float:
        return self.bytes_written / self.written if self.written else 0.0


def run_master(
    job: MasterJob,
    *,
    progress: Callable[[Progress], None] | None = None,
    log: Callable[[str], None] | None = None,
    plan: WarpPlan | None = None,
    cancel: threading.Event | None = None,
) -> MasterSummary:
    """Render the job's frames as PNG masters. Cancellable and resumable per frame."""
    say = log or (lambda _: None)
    tile_size = job.source.tile_size
    if tile_size is None:
        raise ValueError("source tiles are not square or not uniform; cannot stitch")

    todo = job.pending_frames()
    job.directory.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(job.directory).free
    need = job.estimated_bytes(len(todo))
    if free < need:
        raise RuntimeError(
            f"{job.directory} has {free / 2**30:.1f} GiB free; {len(todo)} master(s) need "
            f"about {need / 2**30:.1f} GiB. Refusing to start."
        )

    if plan is None:
        say("building warp plan ...")
        plan = WarpPlan.build(
            job.rig,
            job.width,
            job.height,
            tile_size,
            sampler=job.sampler,
            feather_power=job.feather_power,
        )
        say(
            f"plan ready in {plan.build_seconds:.1f} s ({plan.nbytes / 2**20:.0f} MiB, "
            f"{plan.sampler})"
        )
    elif (plan.width, plan.height, plan.tile_size, plan.sampler, plan.feather_power) != (
        job.width,
        job.height,
        tile_size,
        job.sampler,
        job.feather_power,
    ):
        raise ValueError("the supplied warp plan does not match this job")

    indices = list(job.rig.unique_indices)
    total = len(job.frames)
    skipped = total - len(todo)
    started = time.time()
    done = skipped
    written = 0
    bytes_written = 0
    recent: collections.deque[float] = collections.deque(maxlen=20)
    reports: list[verify.Agreement] = []
    stages = StageTimes()

    def decode(frame: int) -> dict[int, U8]:
        return io.load_tiles(job.source, frame, indices, workers=job.decode_workers)

    def write_one(frame: int, image: io.Panorama) -> int:
        return io.write_png_atomically(
            job.frame_path(frame), image, compress_level=job.compress_level
        )

    stop = cancel if cancel is not None else threading.Event()
    flight: collections.deque[concurrent.futures.Future[int]] = collections.deque()
    with (
        _CancelScope(stop),
        concurrent.futures.ThreadPoolExecutor(max_workers=1) as prefetch,
        concurrent.futures.ThreadPoolExecutor(max_workers=job.write_workers) as writers,
    ):
        try:
            for frame, tiles, waited in _prefetch_tiles(prefetch, decode, todo):
                if stop.is_set():
                    raise Cancelled(f"cancelled after {written} of {len(todo)} frame(s)")
                image, report, warp_seconds, gate_seconds = _stitch_one(
                    plan,
                    tiles,
                    threads=job.warp_threads,
                    gate=(done % job.stats_every) == 0,
                    bit_depth=job.bit_depth,
                )
                if report is not None:
                    reports.append(report)
                    if not report.passed:
                        raise GeometryGateFailed(
                            f"frame {frame} failed the overlap-agreement gate:\n{report.report()}"
                        )
                write_started = time.time()
                flight.append(writers.submit(write_one, frame, image))
                # Keep at most `write_workers` encodes outstanding, so the queue cannot
                # grow into a pile of 88 MB frames while the disk falls behind.
                while len(flight) > job.write_workers:
                    bytes_written += flight.popleft().result()
                blocked = time.time() - write_started
                stages.decode_wait += waited
                stages.warp += warp_seconds
                stages.gate += gate_seconds
                stages.write += blocked
                stages.frames += 1
                done += 1
                written += 1
                recent.append(waited + warp_seconds + gate_seconds + blocked)
                if progress:
                    rate = float(np.mean(recent))
                    progress(
                        Progress(done, total, frame, 0, 1, rate, rate * (total - done))
                    )  # fmt: skip
            while flight:
                bytes_written += flight.popleft().result()
        except BaseException:
            # Outstanding writes are frames that are already stitched, so let them land
            # rather than cancelling them: a resume then has no warp to redo. Each write
            # is atomic, so one failing still leaves no partial file behind, and the
            # original exception is the one that propagates.
            for future in flight:
                with contextlib.suppress(BaseException):
                    future.result()
            raise

    peak = memory.PeakMemory(
        stitcher=memory.peak_commit(), stitcher_resident=memory.peak_working_set()
    )
    warnings: list[str] = []
    if peak.measured:
        say(peak.report())
        if peak.exceeds(job.memory_soft_limit):
            warnings.append(peak.warning(job.memory_soft_limit))
            say(f"WARNING: {warnings[-1]}")
    return MasterSummary(
        directory=job.directory,
        frames=total,
        written=written,
        skipped=skipped,
        seconds=time.time() - started,
        plan_seconds=plan.build_seconds,
        bytes_written=bytes_written,
        gate_reports=tuple(reports),
        stages=stages,
        peak=peak,
        warnings=tuple(warnings),
    )
