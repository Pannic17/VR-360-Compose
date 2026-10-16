"""The PNG and EXR master sinks: lossless frames at the source's native density.

The property that matters most is the one tying it to the rest of the system: a master
must be the *same pixels* the delivery path stitches and the `frame` command prints
metrics for. Everything else -- resume, cancellation, the disk precheck -- is the same
discipline the MP4 segments follow, restated one file at a time.
"""

from __future__ import annotations

import pathlib
import subprocess
import threading
import types

import numpy as np
import pytest
from PIL import Image

from conftest import analytic_panorama, make_source_tree, sample_panorama_into_tile
from vr_compose import encode, exr, io, pipeline, source
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


@pytest.fixture(scope="module")
def exr_source(tmp_path_factory: pytest.TempPathFactory) -> source.SourceSet:
    """The same picture as `still_source`, written as linear half EXR tiles (P15)."""
    rig = twenty_file_rig()
    panorama = analytic_panorama(NATIVE, NATIVE // 2)
    root = tmp_path_factory.mktemp("exr_masters")
    make_source_tree(
        root,
        cameras=20,
        stem="S",
        frames=FRAMES,
        size=(TILE, TILE),
        tile_for=lambda camera, _frame: sample_panorama_into_tile(panorama, rig, camera, TILE),
        tile_format="exr",
    )
    return source.scan(root)[0]


def _job(src: source.SourceSet, directory: pathlib.Path, **overrides: object) -> pipeline.MasterJob:
    # harmonise=False for speed (see tests/test_pipeline.py); the default is covered by
    # test_harmonised_masters_equal_the_frame_command below.
    options: dict[str, object] = dict(stats_every=2, width=NATIVE, harmonise=False)
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
    # the comparison is with `stitch_frame`, the uncorrected reference (harmonise=False is
    # the helper's default); the default (on) is covered by the next test
    job = _job(still_source, tmp_path / "out", compress_level=6)
    summary = pipeline.run_master(job)
    assert summary.written == len(FRAMES) and summary.harmonised is False

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


def _gpu_usable() -> bool:
    from vr_compose import device

    found = device.probe_cuda()
    return found.importable and not found.error and found.count > 0


@pytest.mark.skipif(not _gpu_usable(), reason="no usable CUDA device")
@pytest.mark.parametrize("bit_depth", [8, 16])
def test_cuda_masters_are_the_cpu_masters_byte_for_byte(
    still_source: source.SourceSet,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    bit_depth: int,
) -> None:
    """Metric D through the PNG sink, on the device, at both depths."""
    from vr_compose import device

    monkeypatch.setattr(device, "MIN_CUDA_MEMORY_BYTES", 1)
    cuda = pipeline.run_master(
        _job(still_source, tmp_path / "cuda", device="cuda", bit_depth=bit_depth)
    )
    cpu = pipeline.run_master(_job(still_source, tmp_path / "cpu", bit_depth=bit_depth))
    assert cuda.device == "cuda" and cuda.warnings == ()
    assert cuda.written == cpu.written == len(FRAMES)
    for frame in FRAMES:
        name = f"S_S_{frame:04d}.png"
        assert (tmp_path / "cuda" / name).read_bytes() == (tmp_path / "cpu" / name).read_bytes()


@pytest.mark.parametrize("bit_depth", [8, 16])
def test_exr_masters_track_the_png_masters_and_resume_byte_for_byte(
    still_source: source.SourceSet,
    exr_source: source.SourceSet,
    tmp_path: pathlib.Path,
    bit_depth: int,
) -> None:
    """P15 step 1: an EXR source goes through the same sink. Within one level of the PNG
    source of the same picture (half precision, no dither), and a resumed frame is the
    first run's bytes."""
    assert exr_source.tile_format == "exr" and still_source.tile_format == "png"
    png = pipeline.run_master(_job(still_source, tmp_path / "png", bit_depth=bit_depth))
    first = pipeline.run_master(_job(exr_source, tmp_path / "exr", bit_depth=bit_depth))
    assert png.written == first.written == len(FRAMES)
    for frame in FRAMES:
        name = f"S_S_{frame:04d}.png"
        with Image.open(tmp_path / "png" / name) as image:
            expected = np.asarray(image.convert("RGB")).astype(int)
        with Image.open(tmp_path / "exr" / name) as image:
            got = np.asarray(image.convert("RGB")).astype(int)  # the high byte at 16 bits
        assert np.abs(got - expected).max() <= 1, f"frame {frame}"

    target = tmp_path / "exr" / f"S_S_{FRAMES[2]:04d}.png"
    before = target.read_bytes()
    target.unlink()
    again = pipeline.run_master(_job(exr_source, tmp_path / "exr", bit_depth=bit_depth))
    assert again.written == 1 and again.skipped == len(FRAMES) - 1
    assert target.read_bytes() == before


@pytest.mark.skipif(not _gpu_usable(), reason="no usable CUDA device")
def test_cuda_masters_from_an_exr_source_are_the_cpu_masters_byte_for_byte(
    exr_source: source.SourceSet, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vr_compose import device

    monkeypatch.setattr(device, "MIN_CUDA_MEMORY_BYTES", 1)
    cuda = pipeline.run_master(_job(exr_source, tmp_path / "cuda", device="cuda", bit_depth=16))
    cpu = pipeline.run_master(_job(exr_source, tmp_path / "cpu", bit_depth=16))
    assert cuda.device == "cuda" and cuda.warnings == ()
    assert cuda.written == cpu.written == len(FRAMES)
    for frame in FRAMES:
        name = f"S_S_{frame:04d}.png"
        assert (tmp_path / "cuda" / name).read_bytes() == (tmp_path / "cpu" / name).read_bytes()


def test_write_workers_default_follows_the_device(
    still_source: source.SourceSet, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unset, the writer pool is sized for where the warp *runs*: 1 on the CPU (a second
    writer slows the warp), 4 on the GPU. A `cuda` request that fell back asks for the
    CPU's number; an explicit value is used as given."""
    from vr_compose import device

    assert pipeline.default_write_workers("cpu") == 1
    assert pipeline.default_write_workers("cuda") == 4
    asked: list[str] = []

    def record(device_name: str) -> int:
        asked.append(device_name)
        return pipeline.WRITE_WORKERS[device_name]

    monkeypatch.setattr(pipeline, "default_write_workers", record)
    monkeypatch.setattr(device, "probe_cuda", lambda: device.CudaProbe(importable=True, count=0))
    job = _job(still_source, tmp_path / "out", device="cuda")
    assert job.write_workers is None
    pipeline.run_master(job)
    assert asked == ["cpu"], "the fallback device decides, not the requested one"

    asked.clear()
    pipeline.run_master(_job(still_source, tmp_path / "explicit", write_workers=2))
    assert asked == [], "an explicit value bypasses the default"


def test_harmonised_masters_equal_the_frame_command(
    still_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    """With the real default (harmonise on) the master sink and the `frame` command must
    still agree pixel for pixel: the correction is estimated from the same tiles by the
    same code, so both paths see the same grids."""
    from vr_compose.cli import main

    summary = pipeline.run_master(
        _job(still_source, tmp_path / "out", compress_level=6, harmonise=True)
    )
    assert summary.harmonised is True and summary.written == len(FRAMES)
    single = tmp_path / "frame.png"
    argv = ["--source", str(still_source.root), "frame", "--frame", str(FRAMES[0]),
            "--width", str(NATIVE), "--out", str(single), "--compress-level", "6"]  # fmt: skip
    assert main(argv) == 0
    written = tmp_path / "out" / f"S_S_{FRAMES[0]:04d}.png"
    with Image.open(written) as a, Image.open(single) as b:
        assert np.array_equal(np.asarray(a), np.asarray(b))


# -- the EXR master (P15 step 3) ---------------------------------------------------------------


def _exr_job(
    src: source.SourceSet, directory: pathlib.Path, **overrides: object
) -> pipeline.MasterJob:
    return _job(src, directory, out_format="exr", bit_depth=16, **overrides)


def test_exr_masters_are_the_sixteen_bit_stitch_in_linear_half(
    exr_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    """`<stem>_S_<frame>.exr`, holding the 16-bit stitch decoded to linear half through the
    table -- so the file is the stitcher's uint16 output in another encoding, and the same
    bytes the writer produces when called directly at the same zlib level."""
    job = _exr_job(exr_source, tmp_path / "out")
    summary = pipeline.run_master(job)
    assert summary.written == len(FRAMES) and summary.bytes_written > 0
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        f"S_S_{frame:04d}.exr" for frame in FRAMES
    ]
    assert job.frame_path(999).name == "S_S_0999.exr"

    rig = rig_for(exr_source.camera_count)
    tiles = io.load_tiles(exr_source, FRAMES[0], list(rig.unique_indices))
    reference = stitch_frame(tiles, rig, NATIVE, bit_depth=16).image
    expected = exr.half_from_levels16(np.asarray(reference, np.uint16))
    written = job.frame_path(FRAMES[0])
    header = exr.probe(written)
    assert header.channels == ("B", "G", "R"), "RGB only, no alpha"
    assert (header.width, header.height) == (NATIVE, NATIVE // 2)
    assert np.array_equal(exr.load_half(written), expected)

    direct = tmp_path / "direct.exr"
    exr.write_exr(direct, expected, compress_level=job.compress_level)
    assert written.read_bytes() == direct.read_bytes()


def test_exr_masters_resume_byte_for_byte_and_keep_the_part_discipline(
    exr_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    job = _exr_job(exr_source, tmp_path / "out")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "S_S_0002.exr.part").write_bytes(b"half an EXR")
    assert 2 in job.pending_frames(), "a .part must not count as present"
    first = pipeline.run_master(job)
    assert first.written == len(FRAMES) and first.skipped == 0
    assert not list((tmp_path / "out").glob("*.part"))

    target = job.frame_path(FRAMES[2])
    before = target.read_bytes()
    target.unlink()
    again = pipeline.run_master(job)
    assert again.written == 1 and again.skipped == len(FRAMES) - 1
    assert target.read_bytes() == before


def test_an_exr_master_needs_an_exr_source_at_sixteen_bits(
    still_source: source.SourceSet, exr_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    """A PNG source keeps the refusal: nothing in it needs more than a 16-bit PNG master."""
    with pytest.raises(ValueError, match="EXR source only"):
        _exr_job(still_source, tmp_path / "out")
    with pytest.raises(ValueError, match="16-bit stitch"):
        _job(exr_source, tmp_path / "out", out_format="exr", bit_depth=8)
    with pytest.raises(ValueError, match="out_format"):
        _job(exr_source, tmp_path / "out", out_format="tiff")


def test_the_exr_estimate_has_its_own_rate(
    exr_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    job = _exr_job(exr_source, tmp_path / "out")
    rate = pipeline.EXR_MASTER_BYTES_PER_PIXEL
    assert job.estimated_bytes() == int(5 * NATIVE * (NATIVE // 2) * rate)
    assert job.estimated_bytes(2) == int(2 * NATIVE * (NATIVE // 2) * rate)
    assert rate > 2.83, "must not under-reserve: real 8K EXR masters measure 2.83 bytes/pixel"
    raw = _exr_job(exr_source, tmp_path / "raw", compress_level=0)
    assert raw.estimated_bytes() == int(5 * NATIVE * (NATIVE // 2) * 6.0), "level 0: raw halves"


def test_compress_level_changes_the_exr_bytes_and_not_the_bits(
    exr_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    sizes = {}
    bits = None
    for level in (0, 1, 6):
        summary = pipeline.run_master(
            _exr_job(exr_source, tmp_path / f"l{level}", compress_level=level)
        )
        sizes[level] = summary.bytes_written
        read = exr.load_half(tmp_path / f"l{level}" / f"S_S_{FRAMES[0]:04d}.exr")
        if bits is None:
            bits = read
        assert np.array_equal(read, bits), f"zlib level {level} changed a sample"
    assert sizes[6] <= sizes[1] < sizes[0]


def _ffmpeg() -> encode.Tools | None:
    try:
        return encode.find_tools()
    except encode.FfmpegNotFound:
        return None


@pytest.mark.skipif(_ffmpeg() is None, reason="ffmpeg not available")
def test_ffmpeg_decodes_an_exr_master_to_the_bits_written(
    exr_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    """The independent reading of a master as the sink writes it."""
    tools = _ffmpeg()
    assert tools is not None
    job = _exr_job(exr_source, tmp_path / "out")
    pipeline.run_master(job)
    path = job.frame_path(FRAMES[0])
    command = [str(tools.ffmpeg), "-v", "error", "-i", str(path), "-f", "rawvideo",
               "-pix_fmt", "gbrpf16le", "-"]  # fmt: skip
    raw = subprocess.run(command, capture_output=True, check=True).stdout
    g, b, r = np.frombuffer(raw, np.uint16).reshape(3, NATIVE // 2, NATIVE)
    assert np.array_equal(np.stack([r, g, b], axis=-1), exr.load_half(path))


@pytest.mark.skipif(not _gpu_usable(), reason="no usable CUDA device")
def test_cuda_exr_masters_are_the_cpu_exr_masters_byte_for_byte(
    exr_source: source.SourceSet, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vr_compose import device

    monkeypatch.setattr(device, "MIN_CUDA_MEMORY_BYTES", 1)
    cuda = pipeline.run_master(_exr_job(exr_source, tmp_path / "cuda", device="cuda"))
    cpu = pipeline.run_master(_exr_job(exr_source, tmp_path / "cpu"))
    assert cuda.device == "cuda" and cuda.warnings == ()
    assert cuda.written == cpu.written == len(FRAMES)
    for frame in FRAMES:
        name = f"S_S_{frame:04d}.exr"
        assert (tmp_path / "cuda" / name).read_bytes() == (tmp_path / "cpu" / name).read_bytes()
