"""Frame interpolation in the delivery path (ROADMAP P16 step 3).

The interpolator itself is an external tool and a neural network; nothing here judges its
pictures -- `tools/interp_probe.py` and a headset did that. What is tested is everything
around it, with a stand-in that returns the mean of its two neighbours: which frames reach
the encoder and in what order, that a kept frame is the stitched frame untouched, that a
batch, a segment and a resumed run all see the same neighbours, what happens at a cut and
at the end, and what a job becomes when the tool is not there.

Everything runs at 256x128 on 64-pixel tiles, as the pipeline tests do.
"""

from __future__ import annotations

import dataclasses
import hashlib
import pathlib
import subprocess
import threading
import types
import wave
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

from conftest import analytic_panorama, make_source_tree, sample_panorama_into_tile
from vr_compose import encode, interpolate, pipeline, source
from vr_compose.audio import AudioSpec
from vr_compose.encode import EncodeSpec
from vr_compose.pipeline import Progress, SequenceJob
from vr_compose.rig import rig_for, twenty_file_rig

U8 = npt.NDArray[np.uint8]

try:
    encode.find_tools()
    HAVE_FFMPEG = True
except encode.FfmpegNotFound:
    HAVE_FFMPEG = False

needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not available")
needs_rife = pytest.mark.skipif(
    interpolate.find() is None,
    reason=f"rife-ncnn-vulkan not beside the program (set {interpolate.TOOL_ENVIRONMENT})",
)

WIDTH, HEIGHT, TILE = 256, 128, 64
FRAMES = tuple(range(1, 71))  # 70 stitched frames: one 60-frame segment and a 10-frame tail


# --- the pieces --------------------------------------------------------------------------


def test_padding_is_twelve_and_six_degrees() -> None:
    assert interpolate.padding(7680) == (256, 128), "what the reviewed clip was made with"
    assert interpolate.padding(4096) == (136, 68)
    assert interpolate.padding(WIDTH) == (8, 4)
    assert interpolate.padding(16) == (8, 4), "never nothing"


def direction_pattern(longitude: npt.NDArray[np.float64], latitude: npt.NDArray[np.float64]) -> U8:
    """Three channels that depend on the direction only, so they are defined past a pole."""
    x = np.cos(latitude) * np.cos(longitude)
    y = np.cos(latitude) * np.sin(longitude)
    z = np.sin(latitude)
    levels = np.stack([x, y, z], axis=-1) * 100.0 + 128.0
    return np.asarray(np.rint(levels), dtype=np.uint8)


def grid(
    width: int, height: int, wrap: int = 0, pole: int = 0
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Pixel-centre longitude and latitude, continued `wrap` columns and `pole` rows out."""
    columns = np.arange(-wrap, width + wrap) + 0.5
    rows = np.arange(-pole, height + pole) + 0.5
    longitude = columns / width * 2.0 * np.pi
    latitude = (0.5 - rows / height) * np.pi
    return np.meshgrid(longitude, latitude)


def test_padding_continues_the_sphere() -> None:
    width, height, wrap, pole = 256, 128, 24, 16
    image = direction_pattern(*grid(width, height))
    padded = interpolate.pad_equirect(image, wrap, pole)
    assert padded.shape == (height + 2 * pole, width + 2 * wrap, 3)
    # Latitude past 90 degrees is the same direction as the far side of the pole, so the
    # analytic pattern on the continued grid is what a correct padding must reproduce.
    np.testing.assert_array_equal(padded, direction_pattern(*grid(width, height, wrap, pole)))


@pytest.mark.parametrize(("wrap", "pole"), [(0, 0), (8, 0), (0, 8), (24, 16)])
def test_cropping_undoes_padding(wrap: int, pole: int) -> None:
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, size=(64, 128, 3), dtype=np.uint8)
    padded = interpolate.pad_equirect(image, wrap, pole)
    np.testing.assert_array_equal(interpolate.crop_equirect(padded, wrap, pole), image)


def test_a_staged_frame_comes_back_as_it_went_in(tmp_path: pathlib.Path) -> None:
    rng = np.random.default_rng(3)
    frame = rng.integers(0, 256, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
    interpolate.stage(frame, tmp_path / "000000.png")
    with Image.open(tmp_path / "000000.png") as staged:
        assert staged.mode == "RGB"
        assert staged.size == (WIDTH + 16, HEIGHT + 8), "8 columns each side, 4 rows each pole"
    np.testing.assert_array_equal(
        interpolate.collect(tmp_path / "000000.png", WIDTH, HEIGHT), frame
    )


def test_staging_refuses_what_the_tool_would_garble(tmp_path: pathlib.Path) -> None:
    """A four-channel PNG is read as grey noise, without an error (ROADMAP P16 step 1)."""
    with pytest.raises(ValueError, match="height x width x 3"):
        interpolate.stage(np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8), tmp_path / "x.png")


def test_a_frame_of_another_size_is_not_collected(tmp_path: pathlib.Path) -> None:
    Image.fromarray(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)).save(tmp_path / "bare.png")
    with pytest.raises(interpolate.InterpolationError, match="not the padded 272x136"):
        interpolate.collect(tmp_path / "bare.png", WIDTH, HEIGHT)


def test_resampling_leaves_a_delivery_size_frame_alone() -> None:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    assert interpolate.resample(frame, WIDTH, HEIGHT) is frame
    master = np.full((2 * HEIGHT, 2 * WIDTH, 3), 90, dtype=np.uint8)
    smaller = interpolate.resample(master, WIDTH, HEIGHT)
    assert smaller.shape == (HEIGHT, WIDTH, 3) and smaller.dtype == np.uint8
    assert int(smaller.min()) == int(smaller.max()) == 90, "a flat master stays flat"


def test_frame_difference_is_in_levels() -> None:
    dark = np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8)
    assert interpolate.frame_difference(dark, dark) == 0.0
    assert interpolate.frame_difference(dark, dark + 25) == pytest.approx(25.0)
    # the threshold sits above the fastest motion measured and below a change of scene
    assert 27.0 < interpolate.CUT_LEVELS < 60.0


def _tool_folder(root: pathlib.Path, *, model: bool = True) -> pathlib.Path:
    root.mkdir(parents=True)
    (root / interpolate._executable()).write_bytes(b"")
    if model:
        (root / interpolate.MODEL).mkdir()
    return root


def test_the_tool_is_found_beside_the_program_or_where_the_variable_says(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(interpolate.TOOL_ENVIRONMENT, raising=False)
    monkeypatch.setattr(encode, "_search_dirs", lambda: [tmp_path / "app"])
    assert interpolate.find() is None

    beside = _tool_folder(tmp_path / "app" / interpolate.TOOL_DIRECTORY)
    found = interpolate.find()
    assert found is not None and found.directory == beside.resolve()
    assert found.model == interpolate.MODEL and "rife-v4.6" in found.describe

    elsewhere = _tool_folder(tmp_path / "elsewhere")
    monkeypatch.setenv(interpolate.TOOL_ENVIRONMENT, str(elsewhere))
    named = interpolate.find()
    assert named is not None and named.directory == elsewhere.resolve(), "the variable wins"

    explicit = _tool_folder(tmp_path / "explicit")
    chosen = interpolate.find(explicit)
    assert chosen is not None and chosen.directory == explicit.resolve()


def test_a_folder_without_the_model_is_not_the_tool(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(interpolate.TOOL_ENVIRONMENT, raising=False)
    monkeypatch.setattr(encode, "_search_dirs", lambda: [tmp_path])
    _tool_folder(tmp_path / interpolate.TOOL_DIRECTORY, model=False)
    assert interpolate.find() is None
    assert interpolate.TOOL_DIRECTORY in interpolate.MISSING


class _FakeProcess:
    """Stands in for the tool's process: finishes after `polls` waits, or never."""

    def __init__(self, polls: int, returncode: int, on_exit: Callable[[], None]) -> None:
        self._polls = polls
        self._code = returncode
        self._on_exit = on_exit
        self.returncode: int | None = None
        self.killed = False
        self.stderr = types.SimpleNamespace(read=lambda: b"vkCreateDevice failed\n")

    def wait(self, timeout: float | None = None) -> int:
        if self.killed:
            self.returncode = -9
            return -9
        if self._polls > 0:
            self._polls -= 1
            raise subprocess.TimeoutExpired("rife", timeout or 0.0)
        self._on_exit()
        self.returncode = self._code
        return self._code

    def kill(self) -> None:
        self.killed = True


def _fake_tool(
    monkeypatch: pytest.MonkeyPatch, *, polls: int = 0, returncode: int = 0, writes: int = 99
) -> list[_FakeProcess]:
    """Replace the process the tool runs in; it writes the first `writes` halfway frames."""
    started: list[_FakeProcess] = []

    def popen(command: list[str], **_kwargs: object) -> _FakeProcess:
        out = pathlib.Path(command[command.index("-o") + 1])

        def finish() -> None:
            count = len(list(pathlib.Path(command[command.index("-i") + 1]).glob("*.png")))
            for index in range(min(count - 1, writes)):
                (out / f"{2 * index + 2:08d}.png").write_bytes(b"png")

        process = _FakeProcess(polls, returncode, finish)
        started.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", popen)
    return started


def _staged(directory: pathlib.Path, count: int) -> pathlib.Path:
    directory.mkdir()
    for index in range(count):
        (directory / f"{index:06d}.png").write_bytes(b"png")
    return directory


def test_the_tool_s_halfway_frames_are_picked_out_of_what_it_writes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """For N inputs it writes 2 N files from 1; the even ones are the frames in between."""
    _fake_tool(monkeypatch, polls=2)
    rife = interpolate.Rife(tmp_path / "tool")
    made = rife.synthesise(_staged(tmp_path / "in", 4), 4, tmp_path / "out", stop=lambda: False)
    assert [path.name for path in made] == ["00000002.png", "00000004.png", "00000006.png"]


def test_a_failed_or_short_run_says_what_the_tool_said(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rife = interpolate.Rife(tmp_path / "tool")
    _fake_tool(monkeypatch, returncode=1, writes=0)
    with pytest.raises(interpolate.InterpolationError, match="vkCreateDevice failed"):
        rife.synthesise(_staged(tmp_path / "a", 3), 3, tmp_path / "out_a", stop=lambda: False)
    # exit 0 with frames missing is the same failure: the count is what was asked for
    _fake_tool(monkeypatch, returncode=0, writes=1)
    with pytest.raises(interpolate.InterpolationError, match="1 of 2 frame"):
        rife.synthesise(_staged(tmp_path / "b", 3), 3, tmp_path / "out_b", stop=lambda: False)


def test_a_stop_kills_the_tool_instead_of_waiting_for_the_batch(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = _fake_tool(monkeypatch, polls=10_000)
    rife = interpolate.Rife(tmp_path / "tool")
    asked = iter([False, False, True])
    with pytest.raises(interpolate.Stopped):
        rife.synthesise(_staged(tmp_path / "in", 3), 3, tmp_path / "out", stop=lambda: next(asked))
    assert started[0].killed


@needs_rife
def test_the_real_tool_passes_its_check_and_repeats_itself(tmp_path: pathlib.Path) -> None:
    """Only where the tool is installed: it starts, returns a frame per pair at the size it
    was given, and gives the same bytes twice (the resume guarantee rests on that)."""
    rife = interpolate.find()
    assert rife is not None
    rife.check()
    staged = tmp_path / "in"
    staged.mkdir()
    panorama = analytic_panorama(WIDTH, HEIGHT)
    for index in range(3):
        interpolate.stage(np.roll(panorama, 4 * index, axis=1), staged / f"{index:06d}.png")
    first = rife.synthesise(staged, 3, tmp_path / "a", stop=lambda: False)
    again = rife.synthesise(staged, 3, tmp_path / "b", stop=lambda: False)
    assert len(first) == 2
    assert [p.read_bytes() for p in first] == [p.read_bytes() for p in again]
    middle = interpolate.collect(first[0], WIDTH, HEIGHT)
    assert middle.shape == (HEIGHT, WIDTH, 3)
    # halfway between a frame and the same frame four columns on is closer to two columns
    # on than to either end -- loose, because the picture is the model's business
    target = np.roll(panorama, 2, axis=1)
    near = interpolate.frame_difference(middle, target)
    assert near < interpolate.frame_difference(middle, panorama)


# --- the job -----------------------------------------------------------------------------


def _fake_source(tmp_path: pathlib.Path) -> source.SourceSet:
    make_source_tree(tmp_path, cameras=20, stem="S", frames=FRAMES, size=(TILE, TILE))
    return source.scan(tmp_path)[0]


def test_an_interpolated_job_counts_in_stitched_frames(tmp_path: pathlib.Path) -> None:
    src = _fake_source(tmp_path)
    plain = SequenceJob(
        src, twenty_file_rig(), EncodeSpec(WIDTH, HEIGHT, "h264", 2000, 30), FRAMES,
        tmp_path / "out.mp4", segment_gops=1,
    )  # fmt: skip
    doubled = SequenceJob(
        src, twenty_file_rig(), EncodeSpec(WIDTH, HEIGHT, "h264", 2000, 60), FRAMES,
        tmp_path / "out.mp4", segment_gops=1, interpolate="2x",
    )  # fmt: skip
    assert (plain.frame_factor, doubled.frame_factor) == (1, 2)
    # a GOP is two seconds of the delivery either way: 60 stitched frames at 30 or at 60 fps
    assert [len(frames) for _, frames in doubled.segments()] == [60, 10]
    assert [len(frames) for _, frames in plain.segments()] == [60, 10]
    # ... and the name says which kind of segment it is, so neither is taken for the other
    assert plain.segment_path(0, FRAMES[:60]).name == "out.0000.1-60.mp4"
    assert doubled.segment_path(0, FRAMES[:60]).name == "out.0000.1-60.x2.mp4"
    # the same running time, so the same bytes of video; the batch's PNGs on top
    assert doubled.estimated_output_bytes() == plain.estimated_output_bytes()
    assert plain.estimated_staging_bytes() == 0
    padded = (WIDTH + 16) * (HEIGHT + 8)
    assert doubled.estimated_staging_bytes() == padded * 6 * (interpolate.BATCH + 1)


def test_the_job_refuses_what_it_cannot_double(tmp_path: pathlib.Path) -> None:
    src = _fake_source(tmp_path)
    with pytest.raises(ValueError, match="interpolate must be one of"):
        SequenceJob(
            src, twenty_file_rig(), EncodeSpec(WIDTH, HEIGHT, "h264", 2000, 60), FRAMES,
            tmp_path / "out.mp4", interpolate="3x",
        )  # fmt: skip
    with pytest.raises(ValueError, match="must be even, got 25 fps"):
        SequenceJob(
            src, twenty_file_rig(), EncodeSpec(WIDTH, HEIGHT, "h264", 2000, 25), FRAMES,
            tmp_path / "out.mp4", interpolate="2x",
        )  # fmt: skip


def test_the_disk_check_counts_the_batch(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = SequenceJob(
        _fake_source(tmp_path), twenty_file_rig(),
        EncodeSpec(7680, 3840, "h264", 200_000, 60), FRAMES, tmp_path / "out.mp4",
        interpolate="2x",
    )  # fmt: skip
    video = job.estimated_output_bytes()
    assert job.estimated_staging_bytes() > 5 * 2**30, "31 padded 8K frames, three times over"
    monkeypatch.setattr("shutil.disk_usage", lambda _: types.SimpleNamespace(free=video + 1))
    with pytest.raises(RuntimeError, match="of interpolation frames at a time"):
        pipeline._check_disk(job)


# --- end to end, with a stand-in interpolator (needs ffmpeg) -----------------------------


@dataclasses.dataclass
class Blend:
    """The mean of the two neighbours, written where the real tool writes its frames."""

    calls: list[int] = dataclasses.field(default_factory=list)
    stop_after: threading.Event | None = None

    @property
    def describe(self) -> str:
        return "the mean of the neighbours (test stand-in)"

    def synthesise(
        self, staged: pathlib.Path, count: int, out: pathlib.Path, *, stop: Callable[[], bool]
    ) -> list[pathlib.Path]:
        self.calls.append(count)
        if self.stop_after is not None:
            self.stop_after.set()
        if stop():
            raise interpolate.Stopped
        out.mkdir(parents=True, exist_ok=True)
        frames = [
            np.asarray(Image.open(staged / f"{index:06d}.png"), dtype=np.uint16)
            for index in range(count)
        ]
        made = []
        for index in range(count - 1):
            mean = ((frames[index] + frames[index + 1] + 1) // 2).astype(np.uint8)
            made.append(out / f"{2 * index + 2:08d}.png")
            Image.fromarray(mean).save(made[-1])
        return made


def _mean(a: U8, b: U8) -> U8:
    return ((a.astype(np.uint16) + b.astype(np.uint16) + 1) // 2).astype(np.uint8)


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


def _job(src: source.SourceSet, out: pathlib.Path, **overrides: object) -> SequenceJob:
    options: dict[str, object] = dict(
        segment_gops=1, stats_every=50, keep_segments=True, harmonise=False, interpolate="2x"
    )
    options.update(overrides)
    frames = options.pop("frames", FRAMES)
    fps = 60 if options["interpolate"] == "2x" else 30
    spec_kwargs = {key: options.pop(key) for key in ("deterministic", "master") if key in options}
    return SequenceJob(
        source=src,
        rig=rig_for(src.camera_count),
        spec=EncodeSpec(WIDTH, HEIGHT, "h264", 2000, fps, **spec_kwargs),  # type: ignore[arg-type]
        frames=frames,  # type: ignore[arg-type]
        output=out,
        **options,  # type: ignore[arg-type]
    )


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def written(monkeypatch: pytest.MonkeyPatch) -> list[U8]:
    """Every frame handed to an encoder, in order, as the encoder was handed it."""
    seen: list[U8] = []
    original = encode.SegmentWriter.write

    def spy(self: encode.SegmentWriter, frame: U8) -> None:
        seen.append(np.array(frame, copy=True))
        original(self, frame)

    monkeypatch.setattr(encode.SegmentWriter, "write", spy)
    return seen


@needs_ffmpeg
def test_an_interpolated_run_delivers_twice_the_frames(
    moving_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    seen: list[Progress] = []
    blend = Blend()
    job = _job(moving_source, tmp_path / "out.mp4", keep_segments=False)
    summary = pipeline.run_sequence(job, progress=seen.append, interpolator=blend)

    assert summary.problems == (), summary.problems
    assert summary.frames == 70 and summary.encoded == 70, "counted in stitched frames"
    assert summary.stream.frames == 140
    assert summary.stream.i_intervals == (120,), "two seconds at 60 fps, joins on GOP boundaries"
    assert summary.stream.b_frames == 0 and summary.stream.projection == "equirectangular"
    assert summary.synthesised == 69 and summary.cuts == 0
    assert summary.interpolation == blend.describe
    assert summary.stages.interpolate > 0.0 and "interpolate" in summary.stages.report()
    assert summary.warnings == ()
    # a batch is BATCH frames and the one after it; the last has nothing after it
    assert blend.calls == [31, 31, 10], blend.calls
    assert [r.passed for r in summary.gate_reports] == [True, True], "frames 0 and 50 sampled"
    assert seen[-1].done == 70 and [p.done for p in seen] == list(range(1, 71))
    assert not job.segment_dir.exists(), "segments and the interpolator's scratch are gone"


@needs_ffmpeg
def test_every_stitched_frame_is_kept_and_the_ones_between_are_between(
    moving_source: source.SourceSet, tmp_path: pathlib.Path, written: list[U8]
) -> None:
    """The whole contract, on the frames the encoder actually received.

    A plain run gives the stitched frames. The interpolated run must hand the encoder each
    of them untouched at the even positions, the stand-in's mean of its two neighbours at
    the odd ones -- across batch boundaries (30) and the segment boundary (60) alike --
    and the last frame twice.
    """
    pipeline.run_sequence(_job(moving_source, tmp_path / "plain.mp4", interpolate="off"))
    stitched = list(written)
    written.clear()
    pipeline.run_sequence(_job(moving_source, tmp_path / "doubled.mp4"), interpolator=Blend())

    assert len(stitched) == 70 and len(written) == 140
    for index, frame in enumerate(stitched):
        np.testing.assert_array_equal(written[2 * index], frame, err_msg=f"kept {index}")
    for index in range(69):
        expected = _mean(stitched[index], stitched[index + 1])
        np.testing.assert_array_equal(written[2 * index + 1], expected, err_msg=f"made {index}")
    np.testing.assert_array_equal(written[139], stitched[69], err_msg="the end repeats")


@needs_ffmpeg
def test_a_resumed_interpolated_run_is_the_same_file(
    moving_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    """The last pair of a segment needs the first frame of the next. Resumed alone, a
    segment stitches that frame itself and gets the same pair -- so the bytes match."""
    job = _job(moving_source, tmp_path / "out.mp4", deterministic=True)
    first = pipeline.run_sequence(job, interpolator=Blend())
    clean = _sha(job.output)
    segments = sorted(job.segment_dir.glob("*.mp4"))
    assert [p.name for p in segments] == ["out.0000.1-60.x2.mp4", "out.0001.61-70.x2.mp4"]
    assert not (job.segment_dir / "interpolate").exists(), "kept segments, never the scratch"

    # the first segment is lost: it has to be redone without the second one's help
    job.output.unlink()
    segments[0].unlink()
    blend = Blend()
    resumed = pipeline.run_sequence(job, interpolator=blend)
    assert resumed.skipped_segments == 1 and resumed.encoded == 60
    assert blend.calls == [31, 31], "frame 61 was stitched again, as the look-ahead"
    assert resumed.problems == () and first.problems == ()
    assert resumed.stream.frames == 140
    assert _sha(job.output) == clean


@needs_ffmpeg
def test_plain_segments_are_not_taken_for_interpolated_ones(
    moving_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    out = tmp_path / "out.mp4"
    pipeline.run_sequence(_job(moving_source, out, interpolate="off"))
    out.unlink()
    doubled = pipeline.run_sequence(_job(moving_source, out), interpolator=Blend())
    assert doubled.skipped_segments == 0 and doubled.stream.frames == 140


@needs_ffmpeg
def test_a_cancel_while_the_tool_runs_leaves_nothing_half_done(
    moving_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    job = _job(moving_source, tmp_path / "out.mp4")
    stop = threading.Event()
    with pytest.raises(pipeline.Cancelled, match="cancelled after 0 of 70"):
        pipeline.run_sequence(job, cancel=stop, interpolator=Blend(stop_after=stop))
    assert not list(job.segment_dir.glob("*.mp4")) and not list(job.segment_dir.glob("*.part"))
    assert not list((job.segment_dir / "interpolate").glob("*")), "the batch's PNGs are gone"
    assert not job.output.exists()

    resumed = pipeline.run_sequence(job, interpolator=Blend())
    assert resumed.problems == () and resumed.stream.frames == 140


@needs_ffmpeg
def test_a_cut_is_repeated_not_interpolated(tmp_path: pathlib.Path, written: list[U8]) -> None:
    """Frames 1-6 are one picture, 7-12 a much brighter one: nothing lies in between."""
    rig = twenty_file_rig()
    dark = (analytic_panorama(WIDTH, HEIGHT) // 4).astype(np.uint8)
    bright = (255 - analytic_panorama(WIDTH, HEIGHT) // 4).astype(np.uint8)
    assert interpolate.frame_difference(dark, bright) > interpolate.CUT_LEVELS

    def tile_for(camera: int, frame: int) -> np.ndarray:
        return sample_panorama_into_tile(dark if frame <= 6 else bright, rig, camera, TILE)

    frames = tuple(range(1, 13))
    make_source_tree(
        tmp_path / "src", cameras=20, stem="S", frames=frames, size=(TILE, TILE),
        tile_for=tile_for,
    )  # fmt: skip
    said: list[str] = []
    job = _job(source.scan(tmp_path / "src")[0], tmp_path / "out.mp4", frames=frames, gate="off")
    summary = pipeline.run_sequence(job, log=said.append, interpolator=Blend())

    assert summary.problems == () and summary.stream.frames == 24
    assert summary.cuts == 1 and summary.synthesised == 10, "11 pairs, one of them the cut"
    np.testing.assert_array_equal(written[11], written[10], err_msg="frame 6 shown twice")
    assert not np.array_equal(written[12], written[11]), "then frame 7, untouched"
    assert any("frames 6 and 7" in line and "repeating" in line for line in said), said


def _without_the_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(interpolate, "find", lambda explicit=None: None)


@needs_ffmpeg
def test_without_the_tool_the_job_delivers_the_stitched_frames_and_says_so(
    moving_source: source.SourceSet,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    written: list[U8],
) -> None:
    """Warn, do not block -- and do not fake it: 70 frames at 30 fps, not 140 repeated."""
    _without_the_tool(monkeypatch)
    said: list[str] = []
    job = _job(moving_source, tmp_path / "out.mp4")
    summary = pipeline.run_sequence(job, log=said.append)

    assert summary.problems == (), summary.problems
    assert summary.stream.frames == 70 and len(written) == 70
    assert summary.stream.i_intervals == (60,), "a 30 fps delivery's GOP"
    assert summary.interpolation == "" and summary.synthesised == 0
    assert len(summary.warnings) == 1
    assert "--interpolate 2x" in summary.warnings[0] and "at 30 fps" in summary.warnings[0]
    assert interpolate.TOOL_ENVIRONMENT in summary.warnings[0], "says how to fix it"
    assert any(line.startswith("WARNING: frame interpolation requested") for line in said)
    assert [p.name for p in sorted(job.segment_dir.glob("*.mp4"))] == [
        "out.0000.1-60.mp4",
        "out.0001.61-70.mp4",
    ], "plain segments: it is the plain job now"


@needs_ffmpeg
def test_a_tool_that_does_not_start_is_the_same_warning(
    moving_source: source.SourceSet, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Broken:
        describe = "rife-ncnn-vulkan rife-v4.6 (somewhere)"

        def check(self) -> None:
            raise interpolate.InterpolationError("no Vulkan device")

    monkeypatch.setattr(interpolate, "find", lambda explicit=None: Broken())
    summary = pipeline.run_sequence(_job(moving_source, tmp_path / "out.mp4"))
    assert summary.stream.frames == 70 and summary.problems == ()
    assert "does not run on this machine [no Vulkan device]" in summary.warnings[0]


@needs_ffmpeg
def test_a_larger_master_is_brought_down_before_it_is_interpolated(
    moving_source: source.SourceSet, tmp_path: pathlib.Path, written: list[U8]
) -> None:
    """Stitched at 512x256, delivered at 256x128: both kinds of frame reach the encoder at
    the delivery size, because they were resampled before the interpolator saw them."""
    job = _job(moving_source, tmp_path / "out.mp4", master=(512, 256), frames=FRAMES[:12])
    summary = pipeline.run_sequence(job, interpolator=Blend())
    assert summary.problems == (), summary.problems
    assert (summary.stream.width, summary.stream.height) == (WIDTH, HEIGHT)
    assert summary.stream.frames == 24
    assert {frame.shape for frame in written} == {(HEIGHT, WIDTH, 3)}


def _tone(path: pathlib.Path, seconds: float) -> pathlib.Path:
    samples = (np.sin(np.arange(int(seconds * 48_000)) * 0.05) * 8000).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48_000)
        handle.writeframes(samples.tobytes())
    return path


@needs_ffmpeg
def test_the_audio_is_fitted_to_the_running_time_not_the_frame_count(
    moving_source: source.SourceSet, tmp_path: pathlib.Path
) -> None:
    """70 stitched frames are 2.333 s at 30 fps and still 2.333 s as 140 at 60."""
    job = _job(moving_source, tmp_path / "out.mp4", audio=AudioSpec(_tone(tmp_path / "a.wav", 5)))
    summary = pipeline.run_sequence(job, interpolator=Blend())
    assert summary.problems == (), summary.problems
    assert summary.stream.frames == 140 and summary.stream.audio_streams == 1
    assert summary.stream.audio_duration == pytest.approx(70 / 30, abs=0.03)


# --- a tool that returns something else: the two guards (ROADMAP P16 step 3) -------------
#
# Short of video memory rife-ncnn-vulkan does not fail. With 8 GiB of the card held by
# another allocation it returned 86 of 89 frames that were other pictures entirely, 85
# levels from what they should have been, and exited with 0. Nothing but the frames
# themselves can tell, so they are checked: once at the job's size before anything is
# stitched, and every one of them before it is delivered.


def test_a_frame_between_its_neighbours_passes_and_another_picture_does_not() -> None:
    before = np.full((HEIGHT, WIDTH, 3), 60, dtype=np.uint8)
    after = before + 20
    assert interpolate.not_between(before, before + 10, after) is None
    # a still scene: all three differences are noise, and noise is not a wrong frame
    assert interpolate.not_between(before, before + 3, before) is None
    rng = np.random.default_rng(5)
    noise = rng.integers(0, 256, size=before.shape, dtype=np.uint8)
    reason = interpolate.not_between(before, noise, after)
    assert reason is not None and "which are 20 apart" in reason
    # the margin is far below what a wrong frame measured, and above a still scene's noise
    assert 3.0 < interpolate.BETWEEN_MARGIN < 40.0


class Noise(Blend):
    """What the real tool did with no memory to spare: a frame, the right size, of nothing."""

    def synthesise(
        self, staged: pathlib.Path, count: int, out: pathlib.Path, *, stop: Callable[[], bool]
    ) -> list[pathlib.Path]:
        made = super().synthesise(staged, count, out, stop=stop)
        rng = np.random.default_rng(7)
        for path in made:
            with Image.open(path) as image:
                size = (image.height, image.width, 3)
            Image.fromarray(rng.integers(0, 256, size=size, dtype=np.uint8)).save(path)
        return made


@needs_ffmpeg
def test_a_wrong_frame_stops_the_run_instead_of_being_delivered(
    moving_source: source.SourceSet, tmp_path: pathlib.Path, written: list[U8]
) -> None:
    job = _job(moving_source, tmp_path / "out.mp4")
    with pytest.raises(interpolate.InterpolationError) as caught:
        pipeline.run_sequence(job, interpolator=Noise())
    message = str(caught.value)
    assert "synthesised between 1 and 2 is not a frame between them" in message
    assert "too little free memory" in message and "finished segments are kept" in message
    assert written == [], "not one frame of that batch reached the encoder"
    assert not list(job.segment_dir.glob("*.part")) and not job.output.exists()
    assert not list((job.segment_dir / "interpolate").glob("*"))

    resumed = pipeline.run_sequence(job, interpolator=Blend())
    assert resumed.problems == () and resumed.stream.frames == 140


class _StandInRife(interpolate.Rife):
    """The real class -- so the pipeline rehearses it -- with a stand-in doing the frames."""

    worker: type[Blend] = Blend

    def check(self) -> None:
        return None

    def synthesise(
        self, staged: pathlib.Path, count: int, out: pathlib.Path, *, stop: Callable[[], bool]
    ) -> list[pathlib.Path]:
        return self.worker().synthesise(staged, count, out, stop=stop)


class _StarvedRife(_StandInRife):
    worker = Noise


def test_the_rehearsal_tells_a_working_tool_from_a_starved_one(tmp_path: pathlib.Path) -> None:
    _StandInRife(tmp_path).rehearse(WIDTH, HEIGHT)
    with pytest.raises(interpolate.InterpolationError) as caught:
        _StarvedRife(tmp_path).rehearse(WIDTH, HEIGHT)
    message = str(caught.value)
    assert "a trial frame at 256x128 came back wrong" in message
    assert "too little free memory" in message


@needs_ffmpeg
def test_a_tool_that_fails_its_rehearsal_is_the_fallback_not_a_ruined_delivery(
    moving_source: source.SourceSet, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Found, starts, passes the small check -- and cannot do the job at its size. The run
    warns and delivers the stitched frames at 30 fps, audio fitted to that job."""
    monkeypatch.setattr(interpolate, "find", lambda explicit=None: _StarvedRife(tmp_path))
    job = _job(moving_source, tmp_path / "out.mp4", audio=AudioSpec(_tone(tmp_path / "a.wav", 5)))
    summary = pipeline.run_sequence(job)
    assert summary.problems == (), summary.problems
    assert summary.stream.frames == 70 and summary.stream.i_intervals == (60,)
    assert summary.interpolation == "" and summary.synthesised == 0
    assert len(summary.warnings) == 1 and "cannot do this job here" in summary.warnings[0]
    assert "at 30 fps without it" in summary.warnings[0]
    assert summary.stream.audio_duration == pytest.approx(70 / 30, abs=0.03)


@needs_ffmpeg
def test_a_tool_that_passes_its_rehearsal_runs_the_job(
    moving_source: source.SourceSet, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(interpolate, "find", lambda explicit=None: _StandInRife(tmp_path))
    said: list[str] = []
    job = _job(moving_source, tmp_path / "out.mp4", frames=FRAMES[:12])
    summary = pipeline.run_sequence(job, log=said.append)
    assert summary.problems == () and summary.stream.frames == 24
    assert summary.synthesised == 11 and summary.warnings == ()
    assert any(line.startswith("interpolation rehearsed at 256x128") for line in said), said


def test_the_memory_estimate_is_what_was_measured_at_8k() -> None:
    """6.5 to 7.0 GB on top of the card's baseline for a padded 8K frame (8192x4096)."""
    assert 7.0e9 < interpolate.video_memory_needed(7680, 3840) < 7.6e9
    assert interpolate.video_memory_needed(4096, 2048) < 2.3e9


def test_free_memory_is_unknown_rather_than_zero_without_nvidia_smi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", missing)
    assert interpolate.free_video_memory() is None

    def answer(*_args: object, **_kwargs: object) -> types.SimpleNamespace:
        return types.SimpleNamespace(stdout="11905\n8000\n")

    monkeypatch.setattr(subprocess, "run", answer)
    assert interpolate.free_video_memory() == 11905 * 2**20, "the first card, in bytes"


@needs_ffmpeg
def test_a_card_without_room_is_the_fallback_before_anything_is_stitched(
    moving_source: source.SourceSet, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tool is fine and would pass its trial pair -- that is exactly the case measured:
    the first pairs come back right and the run goes wrong later. Free memory decides."""
    rehearsed: list[tuple[int, int]] = []

    class Counting(_StandInRife):
        def rehearse(self, width: int, height: int) -> None:
            rehearsed.append((width, height))
            super().rehearse(width, height)

    monkeypatch.setattr(interpolate, "find", lambda explicit=None: Counting(tmp_path))
    monkeypatch.setattr(interpolate, "free_video_memory", lambda: 1_000)
    summary = pipeline.run_sequence(_job(moving_source, tmp_path / "short.mp4"))
    assert summary.stream.frames == 70 and summary.problems == ()
    assert "the video card has 0.0 GB free" in summary.warnings[0]
    assert "at 30 fps without it" in summary.warnings[0]
    assert rehearsed == [], "no point trying a pair on a card that has no room"

    # unknown is not "none": without a reading the trial pair is what decides
    monkeypatch.setattr(interpolate, "free_video_memory", lambda: None)
    summary = pipeline.run_sequence(
        _job(moving_source, tmp_path / "unknown.mp4", frames=FRAMES[:12])
    )
    assert summary.stream.frames == 24 and summary.warnings == ()
    assert rehearsed == [(WIDTH, HEIGHT)]
