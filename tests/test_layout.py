"""The layout gate (ROADMAP P11b): camera positions, not pixel values.

Synthetic tiles at 512 px -- the new upstream's size -- sampled from a textured panorama,
so the 128 px patches have something to lock onto. The truncated 20-file set is the case
the gate exists for; the per-camera gain and offset is the case it must *not* react to,
which is what the overlap-agreement gate cannot tell apart (ROADMAP P11's two tables).
"""

from __future__ import annotations

import json
import pathlib
import shutil

import numpy as np
import numpy.typing as npt
import pytest

from conftest import make_source_tree, sample_panorama_into_tile, textured_panorama
from vr_compose import io, layout, pipeline, source, verify
from vr_compose.cli import main
from vr_compose.rig import fifteen_file_rig, rig_for, twenty_file_rig
from vr_compose.stitch import stitch_frame

U8 = npt.NDArray[np.uint8]
TILE = 512
WIDTH = 2048  # the analysis width for 512 px tiles, and their native density


@pytest.fixture(scope="module")
def twenty() -> dict[int, U8]:
    rig = twenty_file_rig()
    panorama = textured_panorama(WIDTH, WIDTH // 2)
    return {i: sample_panorama_into_tile(panorama, rig, i, TILE) for i in range(1, 21)}


@pytest.fixture(scope="module")
def correct(twenty: dict[int, U8]) -> dict[int, U8]:
    """The 15-file layout: the files the 20-file rig reads, renumbered 1..15."""
    return {k: twenty[i] for k, i in enumerate(twenty_file_rig().unique_indices, start=1)}


@pytest.fixture(scope="module")
def truncated(twenty: dict[int, U8]) -> dict[int, U8]:
    """`Camera1..15` of the 20-file set: the same directory shape, the wrong layout."""
    return {i: twenty[i] for i in range(1, 16)}


def test_the_pairs_compare_every_camera() -> None:
    """Each up and down camera sits in exactly one pair, each horizon camera in two."""
    for rig in (fifteen_file_rig(), twenty_file_rig()):
        pairs = layout.pairs_for(rig, "all")
        assert len(pairs) == 10
        seen = [i for pair in pairs for i in pair]
        assert sorted(set(seen)) == sorted(rig.unique_indices)
        for index in rig.unique_indices:
            expected = 2 if rig.view_for(index).elevation == 0.0 else 1
            assert seen.count(index) == expected, (rig.name, index)
    assert len(layout.pairs_for(fifteen_file_rig(), "ring")) == 5
    with pytest.raises(ValueError, match="ring"):
        layout.pairs_for(fifteen_file_rig(), "some")


def test_analysis_width_follows_the_tile_and_is_capped() -> None:
    rig = fifteen_file_rig()
    assert layout.analysis_width(rig, 512) == 2048
    assert layout.analysis_width(rig, 1920) == 3840


def test_the_correct_layout_passes(correct: dict[int, U8]) -> None:
    report = layout.check(correct, fifteen_file_rig())
    assert report.verdict == "PASS", report.report()
    assert report.width == WIDTH
    assert all(p.patches >= layout.MIN_PATCHES for p in report.pairs)
    # nearest-sampled synthetic tiles carry +-1 px of rounding (real ones read 0.00)
    assert report.worst_deg < layout.MAX_SHIFT_DEG / 2, report.report()
    assert "PASS -- 10 of 10 camera pairs checked" in report.summary()


def test_a_truncated_twenty_file_set_fails(truncated: dict[int, U8]) -> None:
    report = layout.check(truncated, fifteen_file_rig())
    assert report.verdict == "FAIL", report.report()
    # sector 0 (cameras 1..3) is numbered the same either way; the next three are not
    failing = {(p.a, p.b) for p in report.failing}
    assert {(5, 6), (4, 6), (8, 9), (7, 9)} <= failing, failing
    assert (2, 3) not in failing and (1, 3) not in failing
    # every mis-numbered pair is well clear of the threshold (1.50-2.37 deg on real tiles)
    assert min(p.median_deg for p in report.failing) > 1.5 * layout.MAX_SHIFT_DEG
    refusal = report.refusal()
    assert "layout gate FAIL" in refusal and "--layout-gate off" in refusal


def test_a_brightness_difference_is_not_a_layout_error(correct: dict[int, U8]) -> None:
    """The reason there are two gates. Per-camera gain and offset -- exposure, fog -- make
    the overlap-agreement gate stop; the layout gate, which looks at positions, passes."""
    rig = fifteen_file_rig()
    shifted = {
        k: np.clip(
            tile.astype(np.float64) * (0.75 if k % 2 else 1.25) + (25 if k % 3 else -25), 0, 255
        ).astype(np.uint8)
        for k, tile in correct.items()
    }
    assert layout.check(shifted, rig).verdict == "PASS"
    agreement = verify.agreement(stitch_frame(shifted, rig, WIDTH).stats)
    assert agreement.fatal, agreement.report()


def test_flat_tiles_cannot_be_checked() -> None:
    flat = {k: np.full((TILE, TILE, 3), 90, np.uint8) for k in range(1, 16)}
    report = layout.check(flat, fifteen_file_rig())
    assert report.verdict == "UNVERIFIABLE"
    assert not report.failing
    assert "too little texture" in report.refusal()


def test_a_disagreeing_pair_beats_an_unchecked_one(truncated: dict[int, U8]) -> None:
    """FAIL wins: one checked pair that disagrees is already an answer."""
    partial = dict(truncated)
    for k in (1, 2, 3):  # sector 0 goes flat, so its two pairs cannot be checked
        partial[k] = np.full((TILE, TILE, 3), 90, np.uint8)
    report = layout.check(partial, fifteen_file_rig())
    assert report.unchecked and report.failing
    assert report.verdict == "FAIL"


def test_the_cropped_measurement_equals_the_full_frame_one(truncated: dict[int, U8]) -> None:
    """The gate warps only each pair's overlap box; the answer must be the full scan's."""
    from vr_compose import projection

    rig = fifteen_file_rig()
    width = 1024  # the full scan is the slow one
    height = width // 2
    dirs = projection.equirect_directions(width, height)
    full = (slice(0, height), slice(0, width))
    for a, b in layout.pairs_for(rig, "all"):
        left, seen_left = layout.warp_luma(truncated[a], rig, a, dirs)
        right, seen_right = layout.warp_luma(truncated[b], rig, b, dirs)
        shared = (seen_left & seen_right).reshape(height, width)
        expected = layout._shifts(
            left.reshape(height, width), right.reshape(height, width), shared, *full, width
        )
        assert sorted(layout.pair_shifts(truncated, rig, a, b, width).shifts) == sorted(expected)


def _write(root: pathlib.Path, tiles: dict[int, U8]) -> pathlib.Path:
    return make_source_tree(
        root,
        cameras=len(tiles),
        stem="Scene",
        frames=[1, 2],
        size=(TILE, TILE),
        tile_for=lambda index, _frame: tiles[index],
        name_digits=2,
    )


def test_frame_refuses_a_wrong_layout_and_writes_nothing(
    tmp_path: pathlib.Path, truncated: dict[int, U8]
) -> None:
    src = _write(tmp_path / "src", truncated)
    out = tmp_path / "out.png"
    with pytest.raises(SystemExit, match="layout gate FAIL") as caught:
        main(["--source", str(src), "frame", "--out", str(out)])
    assert "--layout-gate off" in str(caught.value.code)
    assert not out.exists(), "a refused layout produces no picture"


def test_the_two_gates_switch_off_independently(
    tmp_path: pathlib.Path, truncated: dict[int, U8], capsys: pytest.CaptureFixture[str]
) -> None:
    src = _write(tmp_path / "src", truncated)
    # --gate off leaves the layout gate on
    with pytest.raises(SystemExit, match="layout gate FAIL"):
        main(["--source", str(src), "frame", "--gate", "off", "--out", str(tmp_path / "a.png")])
    # --layout-gate off leaves the overlap-agreement gate on, which stops the truncated set
    code = main(
        ["--source", str(src), "frame", "--layout-gate", "off", "--out", str(tmp_path / "b.png")]
    )
    assert code == 1
    text = capsys.readouterr().out
    assert "layout gate off -- NOT VERIFIED" in text
    assert "verdict    : FAIL" in text
    assert (tmp_path / "b.png").exists(), "the overlap-agreement gate judges, it does not withhold"


def test_frame_reports_a_passing_layout(
    tmp_path: pathlib.Path, correct: dict[int, U8], capsys: pytest.CaptureFixture[str]
) -> None:
    src = _write(tmp_path / "src", correct)
    assert main(["--source", str(src), "frame", "--out", str(tmp_path / "ok.png")]) == 0
    assert "layout     : PASS -- 10 of 10 camera pairs checked" in capsys.readouterr().out


def test_a_twenty_file_set_never_runs_the_layout_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """`1..20` proves the 20-file layout; it must not pay for, or be stopped by, the check."""

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the layout gate ran for a 20-file rig")

    monkeypatch.setattr(layout, "check", refuse)
    assert pipeline.check_layout(None, twenty_file_rig(), 1, "on", print) is None  # type: ignore[arg-type]


def test_the_pipeline_stops_before_creating_its_output(
    tmp_path: pathlib.Path, truncated: dict[int, U8]
) -> None:
    src = source.scan(_write(tmp_path / "src", truncated))[0]
    directory = tmp_path / "masters"
    job = pipeline.MasterJob(
        source=src,
        rig=rig_for(src.camera_count),
        frames=(1, 2),
        directory=directory,
        width=WIDTH,
        harmonise=False,
    )
    lines: list[str] = []
    with pytest.raises(layout.LayoutGateFailed, match="FAIL"):
        pipeline.run_master(job, log=lines.append)
    assert not directory.exists()
    assert any("FAIL -- 10 of 10" in line for line in lines)

    off = pipeline.check_layout(src, job.rig, 1, "off", lines.append)
    assert off is None
    assert lines[-1] == "layout gate off -- the camera numbering is not checked"


def test_a_job_refuses_an_unknown_layout_gate_setting(
    tmp_path: pathlib.Path, correct: dict[int, U8]
) -> None:
    src = source.scan(_write(tmp_path / "src", correct))[0]
    with pytest.raises(ValueError, match="layout_gate"):
        pipeline.MasterJob(
            source=src, rig=fifteen_file_rig(), frames=(1,), directory=tmp_path / "m",
            width=WIDTH, layout_gate="maybe",
        )  # fmt: skip


def test_a_refusal_reaches_the_gui_as_a_layout_error(
    tmp_path: pathlib.Path, truncated: dict[int, U8], capsys: pytest.CaptureFixture[str]
) -> None:
    src = _write(tmp_path / "src", truncated)
    argv = ["--source", str(src), "sequence", "--out-format", "png", "--out", str(tmp_path / "m")]
    with pytest.raises(SystemExit, match="stopped before stitching anything"):
        main([*argv, "--progress-json"])
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    errors = [e for e in events if e["event"] == "error"]
    assert errors and errors[-1]["kind"] == "layout"
    assert not (tmp_path / "m").exists()


@pytest.mark.needs_reference_data
@pytest.mark.slow
def test_the_reference_frame_passes_renumbered_and_fails_truncated(
    reference_root: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """The measurement behind the thresholds, on the real tiles: 0.00 deg against
    1.5-2.4 deg (AGENTS.md section 6). One frame copied, the reference set is read-only."""
    reference = next(s for s in source.scan(reference_root) if s.usable)
    frame = reference.frames[-1]
    verdicts = {}
    for name, keep in (
        ("correct", twenty_file_rig().unique_indices),
        ("truncated", tuple(range(1, 16))),
    ):
        root = tmp_path / name
        for k, i in enumerate(keep, start=1):
            tile = reference.tile_path(i, frame)
            target = root / f"Camera{k:02d}" / "Tempory"
            target.mkdir(parents=True)
            shutil.copyfile(tile, target / tile.name)
        copy = source.scan(root)[0]
        tiles = io.load_tiles(copy, frame, list(range(1, 16)), workers=8)
        verdicts[name] = layout.check(tiles, fifteen_file_rig())
    assert verdicts["correct"].verdict == "PASS"
    assert verdicts["correct"].worst_deg < 0.1
    assert verdicts["truncated"].verdict == "FAIL"
    assert min(p.median_deg for p in verdicts["truncated"].failing) > 1.3
