"""Source-directory detection, entirely against synthetic trees.

The point of these is that detection must not depend on anything specific to the
reference data: not the camera count, not the `Tempory` spelling, not the `L_Cathedral`
stem, not four-digit frame numbers.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from conftest import encode_png, make_source_tree
from vr_compose import source


def test_detects_the_reference_shape(tmp_path: pathlib.Path) -> None:
    make_source_tree(tmp_path, cameras=20, stem="L_Cathedral", frames=range(1656, 1666))
    found = [s for s in source.scan(tmp_path) if s.usable]
    assert len(found) == 1
    detected = found[0]
    assert detected.camera_count == 20
    assert detected.stem == "L_Cathedral"
    assert detected.frames == tuple(range(1656, 1666))
    assert detected.tile_size == 16


@pytest.mark.parametrize(
    ("cameras", "stem", "inner", "digits"),
    [
        (6, "Grotto", "Temporary", 4),
        (4, "Shot_A", None, 3),
        (3, "scene.take2", "frames", 5),
        (2, "X", "a", 2),
    ],
)
def test_detection_is_independent_of_the_reference_specifics(
    tmp_path: pathlib.Path, cameras: int, stem: str, inner: str | None, digits: int
) -> None:
    make_source_tree(
        tmp_path, cameras=cameras, stem=stem, frames=range(1, 4), inner=inner, digits=digits
    )
    found = [s for s in source.scan(tmp_path) if s.usable]
    assert len(found) == 1
    assert found[0].camera_count == cameras
    assert found[0].stem == stem
    assert found[0].frame_digits == digits


def test_camera_directory_naming_variants(tmp_path: pathlib.Path) -> None:
    for name in ("Camera1", "camera_2", "CAMERA 3"):
        directory = tmp_path / name / "Tempory"
        directory.mkdir(parents=True)
        (directory / "S.0001.png").write_bytes(encode_png(np.zeros((8, 8, 3), np.uint8)))
    found = [s for s in source.scan(tmp_path) if s.usable]
    assert len(found) == 1
    assert found[0].camera_count == 3


def test_two_stems_are_two_sets(tmp_path: pathlib.Path) -> None:
    """Two stems are two sets; prefixes are never special-cased."""
    make_source_tree(tmp_path, cameras=4, stem="Shot_A", frames=range(1, 4))
    make_source_tree(tmp_path, cameras=4, stem="Shot_B", frames=range(1, 4))
    found = sorted(s.stem for s in source.scan(tmp_path) if s.usable)
    assert found == ["Shot_A", "Shot_B"]


def test_frames_are_the_intersection(tmp_path: pathlib.Path) -> None:
    make_source_tree(tmp_path, cameras=3, stem="S", frames=[1, 2, 3])
    extra = tmp_path / "Camera2" / "Tempory" / "S.0009.png"
    extra.write_bytes((tmp_path / "Camera2" / "Tempory" / "S.0001.png").read_bytes())
    detected = source.scan(tmp_path)[0]
    assert detected.frames == (1, 2, 3), "only frames every camera has are processable"


def test_ragged_frame_counts_are_a_problem_not_a_silent_subset(
    tmp_path: pathlib.Path,
) -> None:
    make_source_tree(tmp_path, cameras=3, stem="S", frames=[1, 2, 3])
    extra = tmp_path / "Camera2" / "Tempory" / "S.0009.png"
    extra.write_bytes((tmp_path / "Camera2" / "Tempory" / "S.0001.png").read_bytes())
    detected = source.scan(tmp_path)[0]
    assert any("different frame counts" in p for p in detected.problems)
    assert not detected.usable


def test_non_square_tiles_are_refused(tmp_path: pathlib.Path) -> None:
    """The registered rigs assume a square FOV, so an oblong tile is a different layout."""
    make_source_tree(tmp_path, cameras=3, stem="S", frames=[1], size=(24, 16))
    detected = source.scan(tmp_path)[0]
    assert any("not square" in p for p in detected.problems)
    assert detected.tile_size is None
    assert not detected.usable


def test_mixed_resolutions_are_refused(tmp_path: pathlib.Path) -> None:
    make_source_tree(tmp_path, cameras=2, stem="S", frames=[1], size=(16, 16))
    odd = tmp_path / "Camera2" / "Tempory" / "S.0001.png"
    odd.write_bytes(encode_png(np.zeros((32, 32, 3), np.uint8)))
    detected = source.scan(tmp_path)[0]
    assert any("disagree on tile resolution" in p for p in detected.problems)


def test_non_contiguous_camera_numbering_is_refused(tmp_path: pathlib.Path) -> None:
    make_source_tree(tmp_path, cameras=2, stem="S", frames=[1])
    for item in (tmp_path / "Camera2").rglob("*"):
        if item.is_file():
            target = tmp_path / "Camera7" / "Tempory"
            target.mkdir(parents=True, exist_ok=True)
            (target / item.name).write_bytes(item.read_bytes())
    (tmp_path / "Camera2" / "Tempory" / "S.0001.png").unlink()
    (tmp_path / "Camera2" / "Tempory").rmdir()
    (tmp_path / "Camera2").rmdir()
    detected = source.scan(tmp_path)[0]
    assert any("numbering is not" in p for p in detected.problems)


def test_inconsistent_zero_padding_is_refused(tmp_path: pathlib.Path) -> None:
    make_source_tree(tmp_path, cameras=2, stem="S", frames=[1], digits=4)
    odd = tmp_path / "Camera1" / "Tempory" / "S.002.png"
    odd.write_bytes(encode_png(np.zeros((16, 16, 3), np.uint8)))
    detected = source.scan(tmp_path)[0]
    assert any("zero-padded inconsistently" in p for p in detected.problems)


def test_unrelated_directories_yield_nothing(tmp_path: pathlib.Path) -> None:
    (tmp_path / "Camera1").mkdir()
    (tmp_path / "Camera1" / "notes.txt").write_text("not a frame", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    assert source.scan(tmp_path) == []
    assert source.scan(tmp_path / "missing") == []


def test_a_single_camera_is_not_a_source_set(tmp_path: pathlib.Path) -> None:
    make_source_tree(tmp_path, cameras=1, stem="S", frames=[1])
    assert source.scan(tmp_path) == []


def test_output_directory_is_recognised_by_several_names(tmp_path: pathlib.Path) -> None:
    for name in ("FinishTaskOutput", "output", "out"):
        root = tmp_path / name.lower()
        make_source_tree(root, cameras=2, stem="S", frames=[1])
        (root / name).mkdir()
        assert source.scan(root)[0].output_dir == root / name


def test_tiles_deeper_than_one_level_are_not_found(tmp_path: pathlib.Path) -> None:
    """Bounded depth: an unbounded walk would wander into unrelated trees."""
    deep = tmp_path / "Camera1" / "a" / "b"
    deep.mkdir(parents=True)
    (deep / "S.0001.png").write_bytes(encode_png(np.zeros((8, 8, 3), np.uint8)))
    (tmp_path / "Camera2").mkdir()
    assert source.scan(tmp_path) == []


def test_tile_path_round_trips(tmp_path: pathlib.Path) -> None:
    make_source_tree(tmp_path, cameras=3, stem="Scene_Set", frames=[7, 8])
    detected = source.scan(tmp_path)[0]
    path = detected.tile_path(2, 8)
    assert path.name == "Scene_Set.0008.png"
    assert path.exists()
    with pytest.raises(ValueError, match="no camera 9"):
        detected.tile_path(9, 8)


def test_png_dimensions_reads_the_header_only(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "t.png"
    path.write_bytes(encode_png(np.zeros((5, 9, 3), np.uint8)))
    assert source.png_dimensions(path) == (9, 5)

    bad = tmp_path / "bad.png"
    bad.write_bytes(b"definitely not a png")
    with pytest.raises(ValueError, match="not a PNG"):
        source.png_dimensions(bad)


def test_discover_reports_where_it_looked(tmp_path: pathlib.Path) -> None:
    found, searched = source.discover(tmp_path / "nothing-here")
    assert found == []
    assert searched == [tmp_path / "nothing-here"]


def test_discover_finds_an_explicit_directory(tmp_path: pathlib.Path) -> None:
    make_source_tree(tmp_path, cameras=4, stem="S", frames=[1, 2])
    found, searched = source.discover(tmp_path)
    assert [s.stem for s in found] == ["S"]
    assert searched == [tmp_path]


def test_candidate_roots_are_unique_and_start_at_the_base() -> None:
    roots = source.candidate_roots()
    assert roots, "there is always at least the base directory"
    assert len(roots) == len({p.resolve() for p in roots})


def test_describe_mentions_every_problem(tmp_path: pathlib.Path) -> None:
    make_source_tree(tmp_path, cameras=2, stem="S", frames=[1], size=(20, 10))
    text = source.scan(tmp_path)[0].describe()
    assert "not square" in text
    assert "cameras    : 2" in text
