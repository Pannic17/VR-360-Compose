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
    options: dict[str, object] = dict(stats_every=2, width=NATIVE)
    options.update(overrides)
    return pipeline.MasterJob(
        source=src,
        rig=rig_for(src.camera_count),
        frames=FRAMES,
        directory=directory,
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

    written = tmp_path / "out" / f"S_S_{FRAMES[0]:04d}.png"
    with Image.open(written) as image:
        assert image.mode == "RGB", "no alpha: the reference product's 253..255 was blend debris"
        assert np.array_equal(np.asarray(image), reference), "pixels must match the stitcher"

    direct = tmp_path / "direct.png"
    io.write_png(direct, np.asarray(reference, np.uint8), compress_level=6)
    assert written.read_bytes() == direct.read_bytes(), "same level, same bytes"


def test_masters_are_named_stem_S_frame(
    still_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    """P5: `<stem>_S_<frame>.png`, keeping the source's zero padding so listings sort."""
    job = _job(still_source, tmp_path / "out")
    pipeline.run_master(job)
    assert sorted(p.name for p in (tmp_path / "out").glob("*.png")) == [
        f"S_S_{frame:04d}.png" for frame in FRAMES
    ]
    assert job.frame_path(3).name == "S_S_0003.png"
    assert job.frame_path(999).name == "S_S_0999.png", "padded, or 999 sorts after 1000"


def test_frame_mode_is_lossless_and_at_the_input_density(
    still_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    """The user's P5 constraint, made structural rather than conventional.

    Frame mode has no codec, no bitrate and no delivery size to apply, and the width is
    refused outright if it is not what the input implies. The "no compression" half is
    about *lossy* compression: zlib is lossless at every level, which the pixel-identity
    assertion below states as an executable fact rather than a claim.
    """
    rig = rig_for(still_source.camera_count)
    tile = still_source.tile_size
    assert tile is not None
    assert rig.native_width(tile) == NATIVE

    for wrong in (NATIVE // 2, NATIVE * 2, NATIVE + 2):
        with pytest.raises(ValueError, match="input's own density"):
            _job(still_source, tmp_path / "out", width=wrong)

    # every zlib level decodes to the same pixels, so none of them is "compression"
    reference: np.ndarray | None = None
    for level in (0, 1, 9):
        directory = tmp_path / f"level{level}"
        pipeline.run_master(_job(still_source, directory, compress_level=level))
        with Image.open(directory / f"S_S_{FRAMES[0]:04d}.png") as image:
            pixels = np.asarray(image)
        if reference is None:
            reference = pixels
        else:
            assert np.array_equal(pixels, reference), f"zlib level {level} changed a pixel"
    assert (tmp_path / "level0").exists()


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
    (directory / "S_S_0002.png.part").write_bytes(b"half a PNG")
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
    assert present == ["S_S_0001.png", "S_S_0002.png"], present
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
    rate = pipeline.MASTER_BYTES_PER_PIXEL[8]
    assert job.estimated_bytes() == int(5 * NATIVE * (NATIVE // 2) * rate)
    assert job.estimated_bytes(2) == int(2 * NATIVE * (NATIVE // 2) * rate)
    assert rate > 1.41, "must not under-reserve: real 8K masters measure 1.41 bytes/pixel"

    # 16 bits roughly doubles the file, and the estimate has to follow or the precheck lies
    deep = _job(still_source, tmp_path / "out", bit_depth=16)
    assert deep.estimated_bytes() > job.estimated_bytes() * 1.9


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


def test_job_validation(still_source: source.SourceSet) -> None:
    """A real source, because the width check reads the tile size off it."""
    common: dict[str, object] = dict(
        source=still_source,
        rig=twenty_file_rig(),
        frames=FRAMES,
        directory=pathlib.Path("x"),
        width=NATIVE,
    )
    with pytest.raises(ValueError, match="no frames"):
        pipeline.MasterJob(**{**common, "frames": ()})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="even and positive"):
        pipeline.MasterJob(**{**common, "width": 255})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="compress_level"):
        pipeline.MasterJob(**{**common, "compress_level": 11})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="write_workers"):
        pipeline.MasterJob(**{**common, "write_workers": 0})  # type: ignore[arg-type]


def test_sixteen_bit_masters_keep_sub_level_precision(
    still_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    """The point of 16 bits with an 8-bit source: what the *blend* produced between levels.

    Pillow cannot write 16-bit RGB PNG, so `io.write_png16` emits it directly; Pillow can
    still read one, by downconverting to the high byte, which is what makes the last
    assertion here a check on the file rather than on our own writer.
    """
    eight = pipeline.run_master(_job(still_source, tmp_path / "eight"))
    sixteen = pipeline.run_master(_job(still_source, tmp_path / "sixteen", bit_depth=16))
    assert eight.written == sixteen.written == len(FRAMES)
    assert sixteen.bytes_written > eight.bytes_written, "twice the depth, more bytes"

    deep = tmp_path / "sixteen" / f"S_S_{FRAMES[0]:04d}.png"
    header = deep.read_bytes()[8:33]
    assert header[4:8] == b"IHDR"
    assert header[16] == 16, "bit depth in the IHDR"
    assert header[17] == 2, "colour type 2 = RGB"

    with Image.open(deep) as image:
        high_byte = np.asarray(image.convert("RGB"))
    with Image.open(tmp_path / "eight" / f"S_S_{FRAMES[0]:04d}.png") as image:
        shallow = np.asarray(image)

    # 257 maps 255 exactly onto 65535, so the two agree to within a rounding step
    assert np.abs(high_byte.astype(int) - shallow.astype(int)).max() <= 1


def test_the_delivery_path_refuses_a_deep_frame() -> None:
    """A 16-bit frame reaching ffmpeg would be a 256x-too-dark video, not an error."""
    with pytest.raises(ValueError, match="8-bit frames"):
        pipeline._delivery_frame(np.zeros((4, 8, 3), np.uint16))


def test_bit_depth_is_validated(still_source: source.SourceSet, tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="bit_depth"):
        _job(still_source, tmp_path / "out", bit_depth=12)
