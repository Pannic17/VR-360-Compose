"""The 15-directory layout (ROADMAP P11a): same viewpoints as the 20-file rig, numbered 1..15.

The claim that carries everything else is that a 15-file set holding the very files the
20-file rig reads stitches to the **same bytes**. Not "close": the two rigs share their
views by construction and process them in the same order, so any difference is a bug.

The synthetic tiles here are perturbed per camera by up to one level, so the duplicates
of the 20-file set differ the way real ones do (AGENTS.md §3: TAA jitter, up to 1 level).
Without that, a test could not tell *which* duplicate was read -- and that is exactly what
made the first, by-hand attempt at this comparison differ in 680 pixels (ROADMAP P11).
"""

from __future__ import annotations

import pathlib
import shutil

import numpy as np
import numpy.typing as npt
import pytest

from conftest import analytic_panorama, make_source_tree, sample_panorama_into_tile
from vr_compose import harmonise, source
from vr_compose.cli import GATE_OFF_NOTE, main
from vr_compose.rig import fifteen_file_rig, twenty_file_rig
from vr_compose.stitch import SAMPLERS, stitch_frame
from vr_compose.warp import WarpPlan

U8 = npt.NDArray[np.uint8]
WIDTH, TILE = 512, 96


def _twenty_tiles(tile: int, width: int) -> dict[int, U8]:
    """All 20 files, each perturbed by its own +-1 pattern so duplicates are not equal."""
    rig = twenty_file_rig()
    panorama = analytic_panorama(width, width // 2)
    tiles: dict[int, U8] = {}
    for index in range(1, rig.file_count + 1):
        clean = sample_panorama_into_tile(panorama, rig, index, tile).astype(np.int16)
        jitter = np.random.default_rng(index).integers(-1, 2, clean.shape, dtype=np.int16)
        tiles[index] = np.clip(clean + jitter, 0, 255).astype(np.uint8)
    return tiles


def _renumbered(twenty: dict[int, U8], keep: tuple[int, ...]) -> dict[int, U8]:
    """The 15-file view of a 20-file set: `keep[k-1]` becomes camera `k`."""
    return {k: twenty[i] for k, i in enumerate(keep, start=1)}


@pytest.fixture(scope="module")
def twenty() -> dict[int, U8]:
    return _twenty_tiles(TILE, WIDTH)


@pytest.mark.parametrize("sampler", list(SAMPLERS))
@pytest.mark.parametrize("seam_band", [1.0, 0.25])
def test_the_files_the_twenty_rig_reads_stitch_to_the_same_bytes_as_fifteen(
    twenty: dict[int, U8], sampler: str, seam_band: float
) -> None:
    rig20, rig15 = twenty_file_rig(), fifteen_file_rig()
    tiles15 = _renumbered(twenty, rig20.unique_indices)
    kwargs = {"sampler": sampler, "seam_band": seam_band}
    expected = stitch_frame(twenty, rig20, WIDTH, **kwargs).image  # type: ignore[arg-type]
    assert np.array_equal(stitch_frame(tiles15, rig15, WIDTH, **kwargs).image, expected)  # type: ignore[arg-type]
    plan20 = WarpPlan.build(rig20, WIDTH, WIDTH // 2, TILE, sampler=sampler, seam_band=seam_band)
    plan15 = WarpPlan.build(rig15, WIDTH, WIDTH // 2, TILE, sampler=sampler, seam_band=seam_band)
    assert np.array_equal(plan15.apply(tiles15).image, plan20.apply(twenty).image)
    assert plan15.fingerprint != plan20.fingerprint, "two rigs, two cached plans"


def test_harmonised_output_is_the_same_bytes_too(twenty: dict[int, U8]) -> None:
    rig20, rig15 = twenty_file_rig(), fifteen_file_rig()
    tiles15 = _renumbered(twenty, rig20.unique_indices)
    images = []
    for rig, tiles in ((rig20, twenty), (rig15, tiles15)):
        grids = harmonise.Harmoniser(rig, TILE).estimate(tiles).grids
        plan = WarpPlan.build(rig, WIDTH, WIDTH // 2, TILE)
        images.append(plan.apply(tiles, corrections=grids).image)
    assert np.array_equal(images[0], images[1])


def test_keeping_the_other_duplicates_is_not_the_same_bytes(twenty: dict[int, U8]) -> None:
    """The 20-file rig reads the *first* camera of each duplicate pair (4, not 5).

    A copy that keeps 5/9/13/17 instead holds the same views rendered by other cameras,
    which differ by a level here and there -- the 680 pixels ROADMAP P11 measured. This
    pins the representative choice, so a change to it shows up here and not as a
    mysterious one-level drift in a re-render of delivered data.
    """
    rig20 = twenty_file_rig()
    assert rig20.unique_indices == (1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19)
    others = tuple(i for i in range(1, 21) if i % 4)
    expected = stitch_frame(twenty, rig20, WIDTH).image
    other = stitch_frame(_renumbered(twenty, others), fifteen_file_rig(), WIDTH).image
    difference = np.abs(other.astype(int) - expected.astype(int))
    assert difference.any()
    assert difference.mean() < 0.5, "the same views: a few levels at most, here and there"


def _write(root: pathlib.Path, tiles: dict[int, U8], *, name_digits: int = 1) -> pathlib.Path:
    return make_source_tree(
        root,
        cameras=len(tiles),
        stem="Scene",
        frames=[7],
        size=(TILE, TILE),
        tile_for=lambda index, _frame: tiles[index],
        name_digits=name_digits,
    )


def _frame(root: pathlib.Path, out: pathlib.Path, *extra: str) -> int:
    """`frame` with the layout gate off: these smooth 96 px tiles have far too little
    texture for it to compare (it would refuse them as UNVERIFIABLE), and what is tested
    here is the stitch and the overlap-agreement gate. The layout gate has its own file,
    `test_layout.py`, with tiles it can read."""
    argv = ["--source", str(root), "frame", "--width", str(WIDTH), "--out", str(out)]
    return main([*argv, "--layout-gate", "off", *extra])


@pytest.mark.parametrize(
    "extra",
    [(), ("--no-harmonise",), ("--sampler", "nearest", "--seam-band", "0.25")],
    ids=["defaults", "no-harmonise", "nearest-band"],
)
def test_frame_writes_the_same_png_for_both_layouts(
    tmp_path: pathlib.Path, twenty: dict[int, U8], extra: tuple[str, ...]
) -> None:
    """Acceptance 1 end to end, on disk: discovery, decode, harmonise, warp, PNG."""
    src20 = _write(tmp_path / "twenty", twenty)
    tiles15 = _renumbered(twenty, twenty_file_rig().unique_indices)
    src15 = _write(tmp_path / "fifteen", tiles15, name_digits=2)
    assert _frame(src20, tmp_path / "20.png", *extra) == 0
    assert _frame(src15, tmp_path / "15.png", *extra) == 0
    assert (tmp_path / "20.png").read_bytes() == (tmp_path / "15.png").read_bytes()


def test_a_fifteen_file_run_with_the_layout_gate_off_says_so(
    tmp_path: pathlib.Path, twenty: dict[int, U8], capsys: pytest.CaptureFixture[str]
) -> None:
    src15 = _write(tmp_path, _renumbered(twenty, twenty_file_rig().unique_indices))
    assert _frame(src15, tmp_path / "out.png") == 0
    text = capsys.readouterr().out
    assert "of3d-15, reading 15 of 15 files" in text
    assert "layout     : layout gate off -- NOT VERIFIED" in text
    assert main(["--source", str(src15), "discover"]) == 0
    assert "the layout gate checks them when a run starts" in capsys.readouterr().out


def test_a_twenty_file_run_has_no_layout_line(
    tmp_path: pathlib.Path, twenty: dict[int, U8], capsys: pytest.CaptureFixture[str]
) -> None:
    assert _frame(_write(tmp_path, twenty), tmp_path / "out.png") == 0
    assert "layout" not in capsys.readouterr().out


def test_a_truncated_twenty_file_set_fails_the_gate_and_gate_off_is_honoured(
    tmp_path: pathlib.Path, twenty: dict[int, U8], capsys: pytest.CaptureFixture[str]
) -> None:
    """`Camera1..15` of a 20-file set reads as the 15-file rig and is mis-stitched.

    With the layout gate off, the overlap-agreement gate still notices -- by a wide
    margin. And `frame --gate off` means it: the numbers are printed, the exit code no
    longer follows them (it used to return 1 regardless).
    """
    truncated = _write(tmp_path / "src", {i: twenty[i] for i in range(1, 16)})
    assert _frame(truncated, tmp_path / "on.png") == 1
    text = capsys.readouterr().out
    assert "verdict    : FAIL" in text
    median = float(next(line for line in text.splitlines() if line.startswith("median")).split()[2])
    assert median > 2 * 2.5, median

    assert _frame(truncated, tmp_path / "off.png", "--gate", "off") == 0
    text = capsys.readouterr().out
    assert "verdict    : FAIL" in text, "the numbers are still printed"
    assert GATE_OFF_NOTE in text
    assert "layout gate off -- NOT VERIFIED" in text


def test_a_twenty_file_set_missing_its_last_camera_is_refused(
    tmp_path: pathlib.Path, twenty: dict[int, U8]
) -> None:
    """1..19 is contiguous, so the numbering passes; no 19-file rig exists, so it stops."""
    src = _write(tmp_path, {i: twenty[i] for i in range(1, 20)})
    with pytest.raises(SystemExit, match="no rig registered for 19 cameras"):
        _frame(src, tmp_path / "out.png")


@pytest.mark.needs_reference_data
@pytest.mark.slow
def test_the_reference_frame_is_byte_identical_as_fifteen_files(
    reference_root: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """Acceptance 1 on real tiles: keep the files the 20-file rig reads, renumber them.

    One frame is copied (the reference set itself is read-only, AGENTS.md §2); the real
    duplicates differ by up to a level, so keeping the other ones would not pass.
    """
    found = [s for s in source.scan(reference_root) if s.usable]
    assert len(found) == 1
    reference = found[0]
    frame = reference.frames[-1]
    copy = tmp_path / "fifteen"
    for k, i in enumerate(twenty_file_rig().unique_indices, start=1):
        tile = reference.tile_path(i, frame)
        target = copy / f"Camera{k:02d}" / "Tempory"
        target.mkdir(parents=True)
        shutil.copyfile(tile, target / tile.name)

    common = ["--frame", str(frame), "--width", "3840"]
    assert (
        main(["--source", str(reference_root), "frame", *common, "--out", str(tmp_path / "20.png")])
        == 0
    )
    assert main(["--source", str(copy), "frame", *common, "--out", str(tmp_path / "15.png")]) == 0
    assert (tmp_path / "20.png").read_bytes() == (tmp_path / "15.png").read_bytes()
