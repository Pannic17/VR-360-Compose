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
