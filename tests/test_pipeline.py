"""The sequence pipeline end to end on a synthetic source, with resume and the gate.

Everything runs at 256x128 on 64-pixel tiles so a full 130-frame job takes seconds. The
properties under test do not depend on size: whole-GOP segments, finished segments being
skipped, a corrupt segment being redone, byte-identical resume where the encoder allows
it (x264 as-is, x265 only with `deterministic=True`), and the geometry gate stopping a
wrong rig before it wastes a run.
"""

from __future__ import annotations

import datetime
import hashlib
import pathlib
import signal
import sys
import threading
import types

import numpy as np
import pytest

from conftest import analytic_panorama, make_source_tree, sample_panorama_into_tile
from vr_compose import encode, pipeline, source, spherical
from vr_compose.encode import EncodeSpec
from vr_compose.pipeline import GeometryGateFailed, Progress, SequenceJob, parse_frames
from vr_compose.rig import Rig, View, rig_for, twenty_file_rig

try:
    encode.find_tools()
    HAVE_FFMPEG = True
except encode.FfmpegNotFound:
    HAVE_FFMPEG = False

needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not available")

WIDTH, HEIGHT, TILE = 256, 128, 64
FRAMES = tuple(range(1, 131))  # 130 frames: two full 60-frame GOPs and a 10-frame tail


# --- pure logic -----------------------------------------------------------------------


def test_parse_frames_forms() -> None:
    available = list(range(1656, 1666))
    assert parse_frames("all", available) == available
    assert parse_frames("", available) == available
    assert parse_frames("1658-1660", available) == [1658, 1659, 1660]
    assert parse_frames("1664-", available) == [1664, 1665]
    assert parse_frames("-1657", available) == [1656, 1657]
    assert parse_frames("1656, 1660,1665", available) == [1656, 1660, 1665]


def test_parse_frames_rejects_what_is_not_there() -> None:
    # An explicit frame that does not exist is an error ...
    with pytest.raises(ValueError, match="not present in every camera"):
        parse_frames("1,2,3", [10, 11])
    # ... while a range is clipped to what exists, and only an empty result is an error.
    assert parse_frames("5-10", [10, 11]) == [10]
    with pytest.raises(ValueError, match="no frames selected"):
        parse_frames("1-3", [10, 11])
    with pytest.raises(ValueError, match="empty range"):
        parse_frames("20-10", [10, 11, 20])
    with pytest.raises(ValueError, match="no frames shared"):
        parse_frames("all", [])


def _fake_source(tmp_path: pathlib.Path, frames: tuple[int, ...] = FRAMES) -> source.SourceSet:
    make_source_tree(tmp_path, cameras=20, stem="S", frames=frames, size=(TILE, TILE))
    return source.scan(tmp_path)[0]


def test_segments_are_whole_gops_from_the_first_frame(tmp_path: pathlib.Path) -> None:
    job = SequenceJob(
        source=_fake_source(tmp_path),
        rig=twenty_file_rig(),
        spec=EncodeSpec(WIDTH, HEIGHT, "h264", 2000, 30),
        frames=FRAMES,
        output=tmp_path / "out.mp4",
        segment_gops=1,
    )
    lengths = [len(frames) for _, frames in job.segments()]
    assert lengths == [60, 60, 10]
    assert job.segment_path(0, FRAMES[:60]).name == "out.0000.1-60.mp4"
    assert job.segment_dir == tmp_path / "out.segments"


def test_job_validation(tmp_path: pathlib.Path) -> None:
    src = _fake_source(tmp_path, frames=(1, 2))
    spec = EncodeSpec(WIDTH, HEIGHT, "h264", 2000, 30)
    with pytest.raises(ValueError, match="mp4"):
        SequenceJob(src, twenty_file_rig(), spec, (1, 2), tmp_path / "out.mkv")
    with pytest.raises(ValueError, match="segment_gops"):
        SequenceJob(src, twenty_file_rig(), spec, (1, 2), tmp_path / "out.mp4", segment_gops=0)
    with pytest.raises(ValueError, match="no frames"):
        SequenceJob(src, twenty_file_rig(), spec, (), tmp_path / "out.mp4")


def test_estimate_scales_with_bitrate_and_length(tmp_path: pathlib.Path) -> None:
    src = _fake_source(tmp_path, frames=(1, 2))
    base = SequenceJob(
        src,
        twenty_file_rig(),
        EncodeSpec(7680, 3840, "h264", 100_000, 30),
        FRAMES,
        tmp_path / "a.mp4",
    )
    double = SequenceJob(
        src,
        twenty_file_rig(),
        EncodeSpec(7680, 3840, "h264", 200_000, 30),
        FRAMES,
        tmp_path / "b.mp4",
    )
    # int() truncation may shave a byte off either side; the relationship is what matters
    assert double.estimated_output_bytes() == pytest.approx(
        2 * base.estimated_output_bytes(), abs=2
    )
    seconds = len(FRAMES) / 30
    assert base.estimated_output_bytes() == pytest.approx(
        100_000 * 1000 / 8 * seconds * 2 * 1.2, abs=2
    )


def test_progress_line_reports_speedup_against_the_baseline() -> None:
    line = Progress(10, 100, 1665, 0, 3, 2.0, 180.0).line()
    assert "1665" in line and "10/100" in line and "23.4x" in line and "3.0 min" in line
    assert "already done" in Progress(60, 100, 60, 0, 2, 0.0, 0.0, skipped=True).line()


# --- end to end (needs ffmpeg) --------------------------------------------------------


@pytest.fixture(scope="module")
def moving_source(tmp_path_factory: pytest.TempPathFactory) -> source.SourceSet:
    """A 20-camera source whose panorama drifts two columns per frame."""
    rig = twenty_file_rig()
    panorama = analytic_panorama(WIDTH, HEIGHT)
    root = tmp_path_factory.mktemp("moving")
    cache: dict[int, np.ndarray] = {}

    def tile_for(camera: int, frame: int) -> np.ndarray:
        if frame not in cache:
            cache[frame] = np.roll(panorama, 2 * frame, axis=1)
        return sample_panorama_into_tile(cache[frame], rig, camera, TILE)

    make_source_tree(
        root, cameras=20, stem="S", frames=FRAMES, size=(TILE, TILE), tile_for=tile_for
    )
    return source.scan(root)[0]


def _job(
    src: source.SourceSet, out: pathlib.Path, codec: str = "h264", **overrides: object
) -> SequenceJob:
    options: dict[str, object] = dict(segment_gops=1, stats_every=50, keep_segments=True)
    options.update(overrides)
    frames = options.pop("frames", FRAMES)
    spec_kwargs = {key: options.pop(key) for key in ("deterministic", "master") if key in options}
    return SequenceJob(
        source=src,
        rig=rig_for(src.camera_count),
        spec=EncodeSpec(WIDTH, HEIGHT, codec, 2000, 30, **spec_kwargs),  # type: ignore[arg-type]
        frames=frames,  # type: ignore[arg-type]
        output=out,
        **options,  # type: ignore[arg-type]
    )


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@needs_ffmpeg
def test_full_run_produces_a_conformant_mp4(
    moving_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    seen: list[Progress] = []
    job = _job(moving_source, tmp_path / "out.mp4", keep_segments=False)
    summary = pipeline.run_sequence(job, progress=seen.append)

    assert summary.frames == 130 and summary.encoded == 130 and summary.skipped_segments == 0
    assert summary.problems == (), summary.problems
    assert summary.stream.frames == 130
    assert summary.stream.i_intervals == (60,), "segment joins must land exactly on GOP boundaries"
    assert summary.stream.b_frames == 0 and summary.stream.audio_streams == 0
    assert summary.output.exists() and not job.segment_dir.exists()
    # The delivery carries its 360 metadata, and it is the *joined* file that carries it:
    # `-c copy` does not bring the boxes across a concat, so annotating segments would
    # produce a file that plays flat.
    assert summary.stream.projection == "equirectangular"
    written = spherical.read(summary.output)
    assert written is not None and written.stereo_mode == spherical.MONOSCOPIC

    assert [r.passed for r in summary.gate_reports] == [True, True, True], (
        "frames 0, 50, 100 sampled"
    )
    assert seen[-1].done == 130
    assert [p.done for p in seen] == sorted(p.done for p in seen)


@needs_ffmpeg
def test_a_master_is_stitched_then_delivered_smaller(
    moving_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    """The warp renders 512x256; ffmpeg delivers 256x128.

    This is the 4K-from-8K path, and the same one a 16K render delivered at 8K takes: the
    master never reaches the disk and is resampled once, inside the encode.
    """
    job = _job(moving_source, tmp_path / "out.mp4", master=(512, 256))
    summary = pipeline.run_sequence(job)

    assert (summary.stream.width, summary.stream.height) == (256, 128), "delivered size"
    assert summary.stream.frames == 130 and summary.problems == (), summary.problems
    assert summary.stream.i_intervals == (60,)
    assert [r.passed for r in summary.gate_reports] == [True, True, True], "gate ran on masters"


@needs_ffmpeg
def test_a_plan_for_the_delivery_size_is_refused_when_a_master_is_asked_for(
    moving_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    """A plan built at the wrong size must not be silently accepted -- it would deliver
    an upscaled, aliased frame that still passes every conformance check."""
    from vr_compose.warp import WarpPlan

    job = _job(moving_source, tmp_path / "out.mp4", master=(512, 256))
    delivery_plan = WarpPlan.build(job.rig, WIDTH, HEIGHT, TILE)
    with pytest.raises(ValueError, match="does not match this job"):
        pipeline.run_sequence(job, plan=delivery_plan)


@needs_ffmpeg
def test_a_run_without_the_metadata_is_reported_not_thrown_away(
    moving_source: source.SourceSet, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """104 bytes must not cost half an hour of rendering.

    If the metadata cannot be written, the joined file is still a valid MP4 -- it just
    plays flat -- and `metadata --write` can fix it afterwards. So this is a conformance
    problem (non-zero exit, file kept), never an exception.
    """

    def refuse(*_args: object, **_kwargs: object) -> int:
        raise spherical.SphericalError("no video track among 0 track(s)")

    monkeypatch.setattr(spherical, "inject", refuse)
    job = _job(moving_source, tmp_path / "bare.mp4", keep_segments=False)
    summary = pipeline.run_sequence(job)

    assert summary.output.exists(), "the render is kept"
    assert summary.stream.frames == 130
    assert summary.stream.projection == ""
    assert any("metadata --write" in problem for problem in summary.problems), summary.problems
    assert any("spherical projection" in problem for problem in summary.problems)


@needs_ffmpeg
def test_no_spherical_asks_for_no_metadata_and_that_is_not_a_problem(
    moving_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    """`--no-spherical` is an escape hatch, so it must not trip its own conformance check."""
    job = _job(moving_source, tmp_path / "flat.mp4", keep_segments=False, spherical=None)
    summary = pipeline.run_sequence(job)

    assert summary.problems == (), summary.problems
    assert summary.stream.projection == ""
    assert spherical.read(summary.output) is None


@needs_ffmpeg
def test_the_memory_soft_limit_warns_and_does_not_stop_the_run(
    moving_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    """A limit of one byte is certain to be exceeded. The run must still succeed.

    "Advisory" is the kind of property that regresses quietly into "fatal", and the cost
    of that regression is a refused render on a machine that could have done the work.
    """
    job = _job(moving_source, tmp_path / "out.mp4", memory_soft_limit=1)
    summary = pipeline.run_sequence(job)

    assert summary.problems == (), "the cap must not become a conformance failure"
    assert summary.stream.frames == 130 and summary.output.exists()
    assert len(summary.warnings) == 1, summary.warnings
    assert "soft limit" in summary.warnings[0] and "not affected" in summary.warnings[0]


@needs_ffmpeg
def test_a_generous_limit_produces_no_warning_but_still_reports(
    moving_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    summary = pipeline.run_sequence(_job(moving_source, tmp_path / "out.mp4"))
    assert summary.warnings == ()
    if sys.platform == "win32":
        assert summary.peak.measured and summary.peak.encoder > 0
        assert "committed" in summary.peak.report()


@needs_ffmpeg
def test_the_encoder_footprint_does_not_grow_with_frame_count(
    moving_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    """Each segment is its own ffmpeg, so the encoder's peak is a per-segment property.

    Only the encoder side is checked here: `peak.stitcher` is this *process's* lifetime
    high-water mark, and under pytest that includes every other test, so it cannot be
    compared between runs. The stitcher side is verified by two real 8K runs of different
    lengths (AGENTS.md section 8).
    """
    if sys.platform != "win32":
        pytest.skip("Windows memory counters")
    short = pipeline.run_sequence(_job(moving_source, tmp_path / "short.mp4", frames=FRAMES[:60]))
    full = pipeline.run_sequence(_job(moving_source, tmp_path / "full.mp4"))
    assert full.frames == 130 and short.frames == 60
    assert full.peak.encoder == pytest.approx(short.peak.encoder, rel=0.5), (
        short.peak.encoder,
        full.peak.encoder,
    )


RESUME_CASES = [
    pytest.param("h264", False, id="h264-default"),
    pytest.param("h264", True, id="h264-deterministic"),
    pytest.param("h265", False, id="h265-default"),
    pytest.param("h265", True, id="h265-deterministic"),
]


@needs_ffmpeg
@pytest.mark.parametrize(("codec", "deterministic"), RESUME_CASES)
def test_resume_skips_finished_segments(
    moving_source: source.SourceSet, tmp_path: pathlib.Path, codec: str, deterministic: bool
) -> None:
    """A resumed job equals a clean one structurally in every mode, and byte for byte when
    the encoder is pinned to its reproducible mode. Without the pin x264 differs on a short
    tail segment (this job ends in a 10-frame one) and x265 differs whenever it frame-threads.
    """
    job = _job(moving_source, tmp_path / "out.mp4", codec=codec, deterministic=deterministic)
    first = pipeline.run_sequence(job)
    clean_hash = _sha(job.output)
    segments = sorted(job.segment_dir.glob("*.mp4"))
    assert len(segments) == 3

    # an interruption during the last segment: the join and that segment are gone
    job.output.unlink()
    segments[-1].unlink()
    resumed = pipeline.run_sequence(job)
    assert resumed.skipped_segments == 2 and resumed.encoded == 10
    assert resumed.problems == () and first.problems == ()
    assert resumed.stream.frames == first.stream.frames == 130
    assert resumed.stream.i_intervals == first.stream.i_intervals == (60,)
    if deterministic:
        assert _sha(job.output) == clean_hash
    resumed_hash = _sha(job.output)

    # nothing left to do: every segment is skipped and the join is rebuilt from the same
    # files, so this is byte-identical to the resumed join whatever the encoder did
    job.output.unlink()
    again = pipeline.run_sequence(job)
    assert again.skipped_segments == 3 and again.encoded == 0
    assert again.stream == resumed.stream
    assert _sha(job.output) == resumed_hash


@needs_ffmpeg
def test_a_corrupt_segment_is_redone_not_trusted(
    moving_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    job = _job(moving_source, tmp_path / "out.mp4", deterministic=True)
    pipeline.run_sequence(job)
    clean_hash = _sha(job.output)
    middle = sorted(job.segment_dir.glob("*.mp4"))[1]
    middle.write_bytes(b"not an mp4 at all")
    job.output.unlink()

    summary = pipeline.run_sequence(job)
    assert summary.skipped_segments == 2 and summary.encoded == 60
    assert summary.problems == ()
    assert _sha(job.output) == clean_hash


@needs_ffmpeg
def test_no_resume_re_encodes_everything(
    moving_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    job = _job(moving_source, tmp_path / "out.mp4")
    pipeline.run_sequence(job)
    fresh = pipeline.run_sequence(_job(moving_source, tmp_path / "out.mp4", resume=False))
    assert fresh.skipped_segments == 0 and fresh.encoded == 130


@needs_ffmpeg
def test_geometry_gate_stops_a_wrong_rig_before_anything_is_written(
    moving_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    good = rig_for(moving_source.camera_count)
    flipped = Rig("flipped", tuple(View(v.yaw, -v.elevation) for v in good.views))
    job = SequenceJob(
        source=moving_source,
        rig=flipped,
        spec=EncodeSpec(WIDTH, HEIGHT, "h264", 2000, 30),
        frames=FRAMES,
        output=tmp_path / "wrong.mp4",
        segment_gops=1,
        stats_every=50,
    )
    with pytest.raises(GeometryGateFailed, match="frame 1 failed"):
        pipeline.run_sequence(job)
    assert not job.output.exists()
    assert not list(job.segment_dir.glob("*.mp4")), "the aborted segment must not be left behind"


class _Cancelled(Exception):
    """An arbitrary exception from the progress callback: covers the generic abort path.

    Cancellation proper is `threading.Event` / Ctrl-C, covered below -- this one exists to
    prove that *any* escape leaves no partial file."""


@needs_ffmpeg
def test_cancel_mid_segment_leaves_no_partial_and_resumes(
    moving_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    job = _job(moving_source, tmp_path / "out.mp4")

    def cancel_at_70(report: Progress) -> None:
        if report.done == 70:
            raise _Cancelled()

    with pytest.raises(_Cancelled):
        pipeline.run_sequence(job, progress=cancel_at_70)

    finished = sorted(job.segment_dir.glob("*.mp4"))
    assert [p.name.split(".")[2] for p in finished] == ["1-60"], (
        "only the finished segment survives"
    )
    assert not list(job.segment_dir.glob("*.part")), "the aborted segment must be removed"
    assert not job.output.exists()

    resumed = pipeline.run_sequence(job)
    assert resumed.skipped_segments == 1 and resumed.encoded == 70
    assert resumed.problems == () and resumed.stream.frames == 130


@needs_ffmpeg
def test_the_cancel_event_stops_at_a_frame_boundary_and_resumes(
    moving_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    """What Ctrl-C now does: a flag the frame loop checks, so no frame is cut in half.

    The signal handler is the only part not exercised here -- a keypress cannot be
    synthesised in-process -- and it does nothing but set this same event.
    """
    job = _job(moving_source, tmp_path / "out.mp4")
    stop = threading.Event()

    def cancel_at_70(report: Progress) -> None:
        if report.done == 70:
            stop.set()

    with pytest.raises(pipeline.Cancelled, match="cancelled after 70 of 130"):
        pipeline.run_sequence(job, progress=cancel_at_70, cancel=stop)

    finished = sorted(job.segment_dir.glob("*.mp4"))
    assert [p.name.split(".")[2] for p in finished] == ["1-60"]
    assert not list(job.segment_dir.glob("*.part"))
    assert not job.output.exists()
    assert not list(tmp_path.glob("*.part")), "concat must not leave one beside the output"

    resumed = pipeline.run_sequence(job)
    assert resumed.skipped_segments == 1 and resumed.encoded == 70
    assert resumed.problems == () and resumed.stream.frames == 130


@needs_ffmpeg
def test_the_sigint_handler_is_put_back(
    moving_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    """Leaving our handler installed would swallow every later Ctrl-C in the process.

    Checked on both exits -- the clean one and the cancelled one -- because they restore
    through different paths.
    """
    before = signal.getsignal(signal.SIGINT)

    pipeline.run_sequence(_job(moving_source, tmp_path / "clean.mp4"))
    assert signal.getsignal(signal.SIGINT) is before

    stop = threading.Event()
    stop.set()
    with pytest.raises(pipeline.Cancelled):
        pipeline.run_sequence(_job(moving_source, tmp_path / "stopped.mp4"), cancel=stop)
    assert signal.getsignal(signal.SIGINT) is before


def test_a_cancel_set_before_the_run_stops_it_immediately(
    moving_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    """No segment is even opened, so nothing has to be cleaned up."""
    stop = threading.Event()
    stop.set()
    job = _job(moving_source, tmp_path / "out.mp4")
    with pytest.raises(pipeline.Cancelled, match="cancelled after 0 of 130"):
        pipeline.run_sequence(job, cancel=stop)
    assert not job.segment_dir.exists() or not list(job.segment_dir.glob("*"))


@needs_ffmpeg
def test_a_dead_encoder_is_a_failure_not_a_cancellation(
    moving_source: source.SourceSet, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killing the segment's ffmpeg is the shape a console Ctrl-C used to take.

    On Windows the interrupt reaches every process on the console, so ffmpeg died first
    and the pipeline reported "ffmpeg exited early" -- a failure, with an empty message,
    indistinguishable from a real encoder crash. `child_creation_flags` now keeps the
    interrupt away from ffmpeg, which leaves this path meaning only what it says: the
    encoder really did die. It must stay a failure, and it must still leave the disk
    resumable.
    """
    writers: list[encode.SegmentWriter] = []
    original = encode.SegmentWriter.__init__

    def spy(
        self: encode.SegmentWriter,
        tools: encode.Tools,
        spec: EncodeSpec,
        path: pathlib.Path,
    ) -> None:
        original(self, tools, spec, path)
        writers.append(self)

    monkeypatch.setattr(encode.SegmentWriter, "__init__", spy)
    job = _job(moving_source, tmp_path / "out.mp4")

    def kill_at_80(report: Progress) -> None:
        if report.done == 80:
            writers[-1]._process.kill()

    with pytest.raises(RuntimeError, match="ffmpeg exited early") as caught:
        pipeline.run_sequence(job, progress=kill_at_80)
    assert not isinstance(caught.value, pipeline.Cancelled), (
        "a dead encoder is a failure, not a cancellation"
    )

    assert [p.name.split(".")[2] for p in sorted(job.segment_dir.glob("*.mp4"))] == ["1-60"]
    assert not list(job.segment_dir.glob("*.part")) and not job.output.exists()
    monkeypatch.undo()
    resumed = pipeline.run_sequence(job)
    assert resumed.skipped_segments == 1 and resumed.stream.frames == 130


def test_default_output_lands_beside_the_program(tmp_path: pathlib.Path) -> None:
    src = _fake_source(tmp_path, frames=(1, 2, 3))
    assert pipeline.default_output_name(src, "0908_1630") == "S_0908_1630.mp4"
    assert pipeline.default_master_dir_name(src, "0908_1630") == "S_0908_1630"
    base = pipeline.default_output_dir()
    assert base.is_dir()
    assert (base / "pyproject.toml").exists(), "from source, the default is the project root"


def test_the_run_stamp_is_month_day_hour_minute() -> None:
    """`MMDD_HHMM`, the width the user asked for on 2026-09-08."""
    stamp = pipeline.run_stamp(datetime.datetime(2026, 9, 8, 16, 30, 5))
    assert stamp == "0908_1630"
    assert ":" not in stamp, "illegal in Windows filenames"
    # Lexical order is chronological within one year, which is as far as a stamp with no
    # year can go. Documented on `run_stamp` rather than papered over.
    assert stamp < pipeline.run_stamp(datetime.datetime(2026, 9, 8, 16, 31, 0))
    assert pipeline.run_stamp() != "", "the no-argument form reads the clock"


def test_the_stamp_drops_the_seconds_so_a_minute_can_collide() -> None:
    """The cost of `MMDD_HHMM`, asserted rather than assumed.

    This is what `unique_path` exists to absorb: two runs in one minute would otherwise
    be handed the same auto-generated name, and for the MP4 path the same segment
    directory with it.
    """
    first = pipeline.run_stamp(datetime.datetime(2026, 9, 8, 16, 30, 5))
    second = pipeline.run_stamp(datetime.datetime(2026, 9, 8, 16, 30, 59))
    assert first == second


def test_unique_path_leaves_a_free_name_alone(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "S_0908_1630.mp4"
    assert pipeline.unique_path(target) == target


def test_unique_path_suffixes_past_whatever_is_there(tmp_path: pathlib.Path) -> None:
    (tmp_path / "S_0908_1630.mp4").write_bytes(b"")
    assert pipeline.unique_path(tmp_path / "S_0908_1630.mp4").name == "S_0908_1630_2.mp4"
    (tmp_path / "S_0908_1630_2.mp4").write_bytes(b"")
    assert pipeline.unique_path(tmp_path / "S_0908_1630.mp4").name == "S_0908_1630_3.mp4"
    # A frame-mode target is a directory, and a directory is just as taken as a file.
    (tmp_path / "S_0908_1630").mkdir()
    assert pipeline.unique_path(tmp_path / "S_0908_1630").name == "S_0908_1630_2"


def test_refuses_to_start_on_a_full_disk(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _job(_fake_source(tmp_path / "src", frames=(1, 2)), tmp_path / "out.mp4")
    monkeypatch.setattr("shutil.disk_usage", lambda _: types.SimpleNamespace(free=1))
    if not HAVE_FFMPEG:
        pytest.skip("ffmpeg not available")
    with pytest.raises(RuntimeError, match="Refusing to start"):
        pipeline.run_sequence(job)
