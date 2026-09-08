"""The PNG master sink: lossless frames at the source's native density.

The property that matters most is the one tying it to the rest of the system: a master
must be the *same pixels* the delivery path stitches and the `frame` command prints
metrics for. Everything else -- resume, cancellation, the disk precheck -- is the same
discipline the MP4 segments follow, restated one file at a time.
"""

from __future__ import annotations

import pathlib
import threading
import types

import numpy as np
import pytest
from PIL import Image

from conftest import analytic_panorama, make_source_tree, sample_panorama_into_tile
from vr_compose import io, pipeline, source
from vr_compose.rig import rig_for, twenty_file_rig
from vr_compose.stitch import stitch_frame
from vr_compose.warp import WarpPlan

TILE = 64
NATIVE = 256  # twenty_file_rig().native_width(64): 4x the tile, 90 degree fov
FRAMES = (1, 2, 3, 4, 5)


@pytest.fixture(scope="module")
def still_source(tmp_path_factory: pytest.TempPathFactory) -> source.SourceSet:
    rig = twenty_file_rig()
    panorama = analytic_panorama(NATIVE, NATIVE // 2)
    root = tmp_path_factory.mktemp("masters")
    make_source_tree(
        root,
        cameras=20,
        stem="S",
        frames=FRAMES,
        size=(TILE, TILE),
        tile_for=lambda camera, _frame: sample_panorama_into_tile(panorama, rig, camera, TILE),
    )
    return source.scan(root)[0]


def _job(src: source.SourceSet, directory: pathlib.Path, **overrides: object) -> pipeline.MasterJob:
    options: dict[str, object] = dict(stats_every=2)
    options.update(overrides)
    return pipeline.MasterJob(
        source=src,
        rig=rig_for(src.camera_count),
        frames=FRAMES,
        directory=directory,
        width=NATIVE,
        **options,  # type: ignore[arg-type]
    )


def test_master_pixels_equal_the_frame_command(
    still_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    """Metric D, extended to the sink: the master is the stitcher's output, unmodified.

    Byte equality of the *files* additionally needs the same compress_level, since zlib
    level changes the encoding and not the image -- so both are asserted, separately.
    """
    summary = pipeline.run_master(_job(still_source, tmp_path / "out", compress_level=6))
    assert summary.written == len(FRAMES)

    rig = rig_for(still_source.camera_count)
    tiles = io.load_tiles(still_source, FRAMES[0], list(rig.unique_indices))
    reference = stitch_frame(tiles, rig, NATIVE).image

    written = tmp_path / "out" / f"S.{FRAMES[0]:04d}.png"
    with Image.open(written) as image:
        assert image.mode == "RGB", "no alpha: the reference product's 253..255 was blend debris"
        assert np.array_equal(np.asarray(image), reference), "pixels must match the stitcher"

    direct = tmp_path / "direct.png"
    io.write_png(direct, reference, compress_level=6)
    assert written.read_bytes() == direct.read_bytes(), "same level, same bytes"


def test_masters_are_named_like_their_source(
    still_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    job = _job(still_source, tmp_path / "out")
    pipeline.run_master(job)
    assert sorted(p.name for p in (tmp_path / "out").glob("*.png")) == [
        f"S.{frame:04d}.png" for frame in FRAMES
    ]
    assert job.frame_path(3).name == "S.0003.png", "the source's own zero padding"


def test_resume_skips_present_frames_without_decoding_them(
    still_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    """A resumed run must not decode and warp work it is about to throw away."""
    job = _job(still_source, tmp_path / "out")
    first = pipeline.run_master(job)
    assert first.written == 5 and first.skipped == 0

    job.frame_path(2).unlink()
    again = pipeline.run_master(job)
    assert again.written == 1 and again.skipped == 4
    assert again.frames == 5
    assert job.pending_frames() == (), "everything is present again"

    fresh = pipeline.run_master(_job(still_source, tmp_path / "out", resume=False))
    assert fresh.written == 5 and fresh.skipped == 0


def test_an_interrupted_write_cannot_be_mistaken_for_a_finished_frame(
    still_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    """The `.part` discipline: only a completely written file answers "done"."""
    job = _job(still_source, tmp_path / "out")
    directory = tmp_path / "out"
    directory.mkdir()
    (directory / "S.0002.png.part").write_bytes(b"half a PNG")
    assert 2 in job.pending_frames(), "a .part must not count as present"

    pipeline.run_master(job)
    assert not list(directory.glob("*.part")), "and the finished run leaves none"
    with Image.open(job.frame_path(2)) as image:
        assert image.size == (NATIVE, NATIVE // 2)


def test_cancellation_keeps_finished_masters_and_resumes(
    still_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    job = _job(still_source, tmp_path / "out")
    stop = threading.Event()

    def cancel_after_two(report: pipeline.Progress) -> None:
        if report.done == 2:
            stop.set()

    with pytest.raises(pipeline.Cancelled):
        pipeline.run_master(job, progress=cancel_after_two, cancel=stop)

    # Writes already submitted are frames already stitched, so they are allowed to land
    # rather than being cancelled -- otherwise a resume would redo their warp.
    present = sorted(p.name for p in (tmp_path / "out").glob("*.png"))
    assert present == ["S.0001.png", "S.0002.png"], present
    assert not list((tmp_path / "out").glob("*.part"))

    resumed = pipeline.run_master(job)
    assert resumed.skipped == 2 and resumed.written == 3


def test_the_disk_precheck_refuses_rather_than_dying_half_way(
    still_source: source.SourceSet, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shutil.disk_usage", lambda _: types.SimpleNamespace(free=1))
    with pytest.raises(RuntimeError, match="Refusing to start"):
        pipeline.run_master(_job(still_source, tmp_path / "out"))
    assert not list((tmp_path / "out").glob("*.png"))


def test_the_estimate_only_counts_what_is_left_to_write(
    still_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    job = _job(still_source, tmp_path / "out")
    rate = pipeline.MASTER_BYTES_PER_PIXEL
    assert job.estimated_bytes() == int(5 * NATIVE * (NATIVE // 2) * rate)
    assert job.estimated_bytes(2) == int(2 * NATIVE * (NATIVE // 2) * rate)
    assert rate > 1.41, "must not under-reserve: real 8K masters measure 1.41 bytes/pixel"


def test_the_gate_runs_on_masters_too(
    still_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    summary = pipeline.run_master(_job(still_source, tmp_path / "out", stats_every=2))
    assert [r.passed for r in summary.gate_reports] == [True, True, True], "frames 0, 2, 4"
    assert summary.bytes_written > 0 and summary.bytes_per_frame > 0
    assert summary.stages.frames == 5


def test_a_plan_of_the_wrong_size_is_refused(
    still_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    job = _job(still_source, tmp_path / "out")
    wrong = WarpPlan.build(job.rig, NATIVE // 2, NATIVE // 4, TILE)
    with pytest.raises(ValueError, match="does not match this job"):
        pipeline.run_master(job, plan=wrong)


def test_job_validation() -> None:
    rig = twenty_file_rig()
    common = dict(rig=rig, frames=FRAMES, directory=pathlib.Path("x"), width=NATIVE)
    with pytest.raises(ValueError, match="no frames"):
        pipeline.MasterJob(source=None, **{**common, "frames": ()})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="even and positive"):
        pipeline.MasterJob(source=None, **{**common, "width": 255})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="compress_level"):
        pipeline.MasterJob(source=None, compress_level=11, **common)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="write_workers"):
        pipeline.MasterJob(source=None, write_workers=0, **common)  # type: ignore[arg-type]
