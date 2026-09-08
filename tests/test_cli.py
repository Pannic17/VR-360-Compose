"""CLI wiring. The GUI in P5 calls these same library functions, so this covers both."""

from __future__ import annotations

import pathlib

import pytest

from conftest import analytic_panorama, make_source_tree, sample_panorama_into_tile
from vr_compose import __version__
from vr_compose.cli import build_parser, main
from vr_compose.rig import twenty_file_rig


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_a_subcommand_is_required(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main([])
    assert "command" in capsys.readouterr().err


def test_parser_exposes_both_subcommands() -> None:
    parser = build_parser()
    for argv in (["discover"], ["frame"]):
        assert parser.parse_args(argv).command == argv[0]


def test_discover_reports_the_set_and_the_rig(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_source_tree(tmp_path, cameras=20, stem="Scene_A", frames=[1, 2])
    assert main(["--source", str(tmp_path), "discover"]) == 0
    out = capsys.readouterr().out
    assert "cameras    : 20" in out
    assert "of3d-20" in out
    assert "15 distinct of 20 files" in out


def test_discover_explains_an_unknown_layout(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_source_tree(tmp_path, cameras=6, stem="S", frames=[1])
    assert main(["--source", str(tmp_path), "discover"]) == 0
    assert "UNKNOWN" in capsys.readouterr().out


def test_discover_lists_the_problems_of_a_rejected_directory(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_source_tree(tmp_path, cameras=4, stem="S", frames=[1], size=(24, 16))
    assert main(["--source", str(tmp_path), "discover"]) == 1
    assert "not square" in capsys.readouterr().out


def test_frame_stitches_and_passes_the_gate(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = twenty_file_rig()
    panorama = analytic_panorama(512, 256)
    tiles = {i: sample_panorama_into_tile(panorama, rig, i, 128) for i in range(1, 21)}
    make_source_tree(
        tmp_path,
        cameras=20,
        stem="Scene_A",
        frames=[5],
        size=(128, 128),
        tile_for=lambda index, _frame: tiles[index],
    )
    out = tmp_path / "out" / "pano.png"
    argv = [
        "--source",
        str(tmp_path),
        "frame",
        "--frame",
        "5",
        "--width",
        "512",
        "--out",
        str(out),
    ]
    assert main(argv) == 0
    text = capsys.readouterr().out
    assert "reading 15 of 20 files" in text
    assert "PASS" in text
    assert out.exists()


def test_frame_rejects_an_absent_frame(tmp_path: pathlib.Path) -> None:
    make_source_tree(tmp_path, cameras=20, stem="S", frames=[1, 2])
    with pytest.raises(SystemExit, match="not present in every camera"):
        main(["--source", str(tmp_path), "frame", "--frame", "99", "--width", "128"])


def test_frame_refuses_an_unknown_rig(tmp_path: pathlib.Path) -> None:
    make_source_tree(tmp_path, cameras=6, stem="S", frames=[1])
    with pytest.raises(SystemExit, match="no rig registered"):
        main(["--source", str(tmp_path), "frame", "--width", "128"])


def test_frame_requires_a_valid_stem_choice(tmp_path: pathlib.Path) -> None:
    make_source_tree(tmp_path, cameras=20, stem="Scene_A", frames=[1])
    make_source_tree(tmp_path, cameras=20, stem="Scene_B", frames=[1])
    with pytest.raises(SystemExit, match="available: "):
        main(["--source", str(tmp_path), "frame", "--stem", "Q_S", "--width", "128"])


def test_missing_source_exits_with_guidance(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--source", str(tmp_path / "nope"), "frame"])
    assert caught.value.code == 2
    assert "CameraN" in capsys.readouterr().err
