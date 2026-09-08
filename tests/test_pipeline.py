"""The sequence pipeline end to end on a synthetic source, with resume and the gate.

Everything runs at 256x128 on 64-pixel tiles so a full 130-frame job takes seconds. The
properties under test do not depend on size: whole-GOP segments, finished segments being
skipped, a corrupt segment being redone, byte-identical resume where the encoder allows
it (x264 as-is, x265 only with `deterministic=True`), and the geometry gate stopping a
wrong rig before it wastes a run.
"""

from __future__ import annotations

import hashlib
import pathlib
import types

import numpy as np
import pytest

from conftest import analytic_panorama, make_source_tree, sample_panorama_into_tile
from vr_compose import encode, pipeline, source
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
    spec_kwargs = {key: options.pop(key) for key in ("deterministic", "master") if key in options}
    return SequenceJob(
        source=src,
        rig=rig_for(src.camera_count),
        spec=EncodeSpec(WIDTH, HEIGHT, codec, 2000, 30, **spec_kwargs),  # type: ignore[arg-type]
        frames=FRAMES,
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
    """Stands in for Ctrl-C: raised from the progress callback mid-segment."""


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


def test_default_output_lands_beside_the_program(tmp_path: pathlib.Path) -> None:
    src = _fake_source(tmp_path, frames=(1, 2, 3))
    spec = EncodeSpec(WIDTH, HEIGHT, "h265", 2000, 30)
    assert pipeline.default_output_name(src, spec, [1, 2, 3]) == "S.1-3.256x128.h265.mp4"
    base = pipeline.default_output_dir()
    assert base.is_dir()
    assert (base / "pyproject.toml").exists(), "from source, the default is the project root"


def test_refuses_to_start_on_a_full_disk(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _job(_fake_source(tmp_path / "src", frames=(1, 2)), tmp_path / "out.mp4")
    monkeypatch.setattr("shutil.disk_usage", lambda _: types.SimpleNamespace(free=1))
    if not HAVE_FFMPEG:
        pytest.skip("ffmpeg not available")
    with pytest.raises(RuntimeError, match="Refusing to start"):
        pipeline.run_sequence(job)
