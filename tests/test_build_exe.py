"""The render check of `tools/build_exe.py --smoke` picks its frames from the source.

It takes the stem and two frames the source itself has, preferring two consecutive ones, so the
check runs on whatever capture is at hand.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

from conftest import make_source_tree

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))

import build_exe


def test_takes_the_first_consecutive_pair(tmp_path: pathlib.Path) -> None:
    # A lone frame before the run, as the reference data carries one, is skipped.
    make_source_tree(tmp_path, cameras=15, stem="scene", frames=[0, 600, 601, 602], name_digits=2)
    assert build_exe.smoke_frames(tmp_path) == ("scene", "600,601")


def test_takes_the_first_two_when_none_are_consecutive(tmp_path: pathlib.Path) -> None:
    make_source_tree(tmp_path, cameras=15, stem="scene", frames=[10, 20, 30], name_digits=2)
    assert build_exe.smoke_frames(tmp_path) == ("scene", "10,20")


def test_skips_a_stem_with_a_single_frame(tmp_path: pathlib.Path) -> None:
    make_source_tree(tmp_path, cameras=15, stem="a_short", frames=[5], name_digits=2)
    make_source_tree(tmp_path, cameras=15, stem="b_long", frames=[7, 8], name_digits=2)
    assert build_exe.smoke_frames(tmp_path) == ("b_long", "7,8")


def test_refuses_a_source_with_no_usable_pair(tmp_path: pathlib.Path) -> None:
    make_source_tree(tmp_path, cameras=15, stem="scene", frames=[5], name_digits=2)
    with pytest.raises(SystemExit, match="two frames"):
        build_exe.smoke_frames(tmp_path)


# --- the interpolator rides along (ROADMAP P16, user's decision 2026-11-11) --------------


def _release(root: pathlib.Path) -> pathlib.Path:
    """The tool's release folder as downloaded: one executable, its library and licence,
    and more models than the product uses."""
    root.mkdir(parents=True)
    for name in (build_exe.RIFE_EXE, "vcomp140.dll", "LICENSE", "README.md"):
        (root / name).write_bytes(name.encode())
    for model in (build_exe.RIFE_MODEL, "rife-v2.3", "rife-anime"):
        (root / model).mkdir()
        for name in ("flownet.bin", "flownet.param"):
            (root / model / name).write_bytes(f"{model}/{name}".encode())
    return root


def test_only_what_the_product_runs_is_shipped(tmp_path: pathlib.Path) -> None:
    release = _release(tmp_path / "release")
    shipped = [(file.name, subfolder) for file, subfolder in build_exe.rife_files(release)]
    assert shipped == [
        (build_exe.RIFE_EXE, ""),
        ("vcomp140.dll", ""),
        ("LICENSE", ""),  # the MIT terms ask for it in every copy
        ("flownet.bin", build_exe.RIFE_MODEL),
        ("flownet.param", build_exe.RIFE_MODEL),
    ], "one model of the three, and no README"

    payload = build_exe.gather_rife(release, tmp_path / "payload")
    laid_out = sorted(p.relative_to(payload).as_posix() for p in payload.rglob("*") if p.is_file())
    assert laid_out == [
        "LICENSE",
        build_exe.RIFE_EXE,
        f"{build_exe.RIFE_MODEL}/flownet.bin",
        f"{build_exe.RIFE_MODEL}/flownet.param",
        "vcomp140.dll",
    ]
    # gathered again, it is the same folder and not the old one with more in it
    (payload / "stale.txt").write_bytes(b"x")
    build_exe.gather_rife(release, payload)
    assert not (payload / "stale.txt").exists()


def test_a_named_folder_must_be_the_tool(tmp_path: pathlib.Path) -> None:
    release = _release(tmp_path / "release")
    assert build_exe.locate_rife(release) == release
    (release / build_exe.RIFE_EXE).unlink()
    with pytest.raises(SystemExit, match="does not hold"):
        build_exe.locate_rife(release)


def test_a_build_that_cannot_find_the_tool_stops_and_says_how_to_go_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vr_compose import interpolate

    monkeypatch.setattr(interpolate, "find", lambda explicit=None: None)
    with pytest.raises(SystemExit) as caught:
        build_exe.locate_rife(None)
    assert "--rife" in str(caught.value) and "--no-rife" in str(caught.value)


def test_the_folder_build_gets_the_tool_beside_the_executable(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(build_exe, "DIST", tmp_path / "dist")
    payload = build_exe.gather_rife(_release(tmp_path / "release"), tmp_path / "payload")

    folder = build_exe.Layout(onefile=False)
    folder.directory.mkdir(parents=True)
    build_exe.stage_rife(folder, payload)
    beside = folder.directory / build_exe.RIFE_FOLDER
    assert (beside / build_exe.RIFE_EXE).is_file()
    assert (beside / build_exe.RIFE_MODEL / "flownet.bin").is_file()

    # one file: nothing beside it, the bundle carries the folder (VRC_RIFE_DIR)
    single = build_exe.Layout(onefile=True)
    single.directory.mkdir(parents=True)
    build_exe.stage_rife(single, payload)
    assert not (single.directory / build_exe.RIFE_FOLDER).exists()
    build_exe.stage_rife(folder, None)
    printed = capsys.readouterr().out
    assert "inside the executable" in printed and "--no-rife" in printed


def test_the_folder_names_are_the_ones_the_program_looks_for() -> None:
    from vr_compose import interpolate

    assert build_exe.RIFE_FOLDER == interpolate.TOOL_DIRECTORY
    assert build_exe.RIFE_MODEL == interpolate.MODEL
    assert build_exe.RIFE_EXE == "rife-ncnn-vulkan.exe"
