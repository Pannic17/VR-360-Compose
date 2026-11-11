"""The interpolation probe (ROADMAP P16): the parts that decide what its numbers mean.

None of this needs an interpolator or real frames: the statistics are checked against
differences placed where each column claims to look. The padding the probe started with
lives in the product now and is tested with it (`tests/test_interpolate.py`).
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))

import interp_probe as probe

U8 = npt.NDArray[np.uint8]
F32 = npt.NDArray[np.float32]


def views_of(level: float) -> tuple[F32, F32]:
    mean = np.full((probe.MEAN_WIDTH // 2, probe.MEAN_WIDTH), level, dtype=np.float32)
    local = np.full((probe.LOCAL_WIDTH // 2, probe.LOCAL_WIDTH), level, dtype=np.float32)
    return mean, local


def columns(stats: probe.Stats) -> dict[str, float]:
    names = stats.HEAD.split()[1:]
    values = stats.line("x").split()[1:]
    return dict(zip(names, map(float, values), strict=True))


def test_identical_frames_score_zero() -> None:
    stats = probe.Stats()
    frame = probe.Views(*views_of(100.0))
    stats.add(0, frame, frame)
    assert all(value == 0.0 for value in columns(stats).values())


def test_each_column_looks_where_it_says() -> None:
    height = probe.LOCAL_WIDTH // 2

    def row_of(latitude: float) -> int:
        return int((0.5 - latitude / 180.0) * height)

    places = {
        "<30": (row_of(10.0), 1000),
        "30-60": (row_of(-45.0), 1000),
        "zenith": (row_of(80.0), 1000),
        "nadir": (row_of(-80.0), 1000),
        "wrap": (row_of(0.0), 5),
    }
    for name, (row, column) in places.items():
        mean, local = views_of(100.0)
        changed = local.copy()
        # 64 rows so the patch survives two printed decimals; 8 columns stay in the strip
        changed[row : row + 64, column : column + (8 if name == "wrap" else 64)] += 50.0
        stats = probe.Stats()
        stats.add(0, probe.Views(mean, changed), probe.Views(mean, local))
        seen = columns(stats)
        bands = {key: seen[key] for key in ("<30", "30-60", "zenith", "nadir")}
        band = name if name in bands else "<30"
        assert bands[band] > 0.0, name
        assert all(value == 0.0 for key, value in bands.items() if key != band), name
        assert (seen["wrap"] > 0.0) == (name == "wrap"), name
        # the caps are outside the |lat| < 60 strip the seam, tail and block columns use
        inside = name not in ("zenith", "nadir")
        assert (seen["blkmax"] > 0.0) == inside, name
        assert seen["mean"] == 0.0, "the mean column reads the other view"


def test_mean_is_weighted_by_solid_angle() -> None:
    mean, _ = views_of(100.0)
    polar, equatorial = mean.copy(), mean.copy()
    polar[:8] += 64.0
    middle = probe.MEAN_WIDTH // 4
    equatorial[middle : middle + 8] += 64.0
    at_pole = probe.weighted_mean(polar, mean)
    at_equator = probe.weighted_mean(equatorial, mean)
    assert at_equator > 50 * at_pole > 0.0
    # eight full rows at the equator are 8 / 1024 of the rows and pi / 2 of the mean weight
    assert at_equator == pytest.approx(64.0 * 8 / (probe.MEAN_WIDTH // 2) * np.pi / 2, rel=1e-3)


def test_worst_block_is_located() -> None:
    mean, local = views_of(100.0)
    changed = local.copy()
    changed[900:964, 1280:1344] += 40.0
    stats = probe.Stats()
    stats.add(7, probe.Views(mean, changed), probe.Views(mean, local))
    worst = stats.worst[0]
    assert (worst.index, worst.column) == (7, 1280)
    assert worst.row <= 900 < worst.row + probe.BLOCK or worst.row == 896 + probe.BLOCK
    assert worst.value > 0.0


def test_methods_parse() -> None:
    assert probe.parse_method("repeat").kind == "repeat"
    assert probe.parse_method("minterpolate:3840").width == 3840
    rife = probe.parse_method("rife:rife-v4.6:pad:x")
    assert (rife.model, rife.pad, rife.flags) == ("rife-v4.6", True, ("-x",))
    assert probe.parse_method("rife:rife-v4.6").pad is False
    for bad in ("rife", "minterpolate", "blend:2", "rife:m:sharp", "morph"):
        with pytest.raises(SystemExit):
            probe.parse_method(bad)


def write(path: pathlib.Path, image: U8) -> pathlib.Path:
    Image.fromarray(image).save(path)
    return path


def test_blend_is_the_rounded_mean_of_neighbours(tmp_path: pathlib.Path) -> None:
    levels = (10, 21, 40)
    kept = [
        write(tmp_path / f"{i:06d}.png", np.full((4, 8, 3), level, dtype=np.uint8))
        for i, level in enumerate(levels)
    ]
    out = tmp_path / "out"
    out.mkdir()
    result = probe.do_blend(kept, out)
    assert sorted(result.frames) == [0, 1]
    assert int(probe.read_rgb(result.frames[0])[0, 0, 0]) == 16  # (10 + 21 + 1) // 2
    assert int(probe.read_rgb(result.frames[1])[0, 0, 0]) == 31  # (21 + 40 + 1) // 2


def test_truth_is_every_other_frame_from_the_second(tmp_path: pathlib.Path) -> None:
    for i in range(7):
        write(tmp_path / f"{i:06d}.png", np.zeros((2, 4, 3), dtype=np.uint8))
    # four kept frames give three synthesised ones, scored against frames 1, 3 and 5
    truth = probe._truth(4, tmp_path)
    assert {i: path.name for i, path in truth.items()} == {
        0: "000001.png",
        1: "000003.png",
        2: "000005.png",
    }
    # a truth directory that stops early limits what can be scored, it does not fail
    assert sorted(probe._truth(9, tmp_path)) == [0, 1, 2]


def test_four_channel_tiles_are_staged_as_rgb(tmp_path: pathlib.Path) -> None:
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[..., :3] = (10, 20, 30)
    rgba[..., 3] = 255
    source_tile = write(tmp_path / "tile.png", rgba)
    probe.stage_rgb(source_tile, tmp_path / "staged.png")
    with Image.open(tmp_path / "staged.png") as staged:
        assert staged.mode == "RGB"
        assert staged.getpixel((0, 0)) == (10, 20, 30)
    plain = write(tmp_path / "plain.png", rgba[..., :3].copy())
    probe.stage_rgb(plain, tmp_path / "linked.png")
    assert (tmp_path / "linked.png").read_bytes() == plain.read_bytes()


def test_a_named_directory_is_never_emptied(tmp_path: pathlib.Path) -> None:
    assert probe.new_directory(tmp_path / "new").is_dir()
    (tmp_path / "new" / "tile.png").write_bytes(b"x")
    with pytest.raises(SystemExit):
        probe.new_directory(tmp_path / "new")
    assert (tmp_path / "new" / "tile.png").exists()


def test_a_clip_keeps_every_frame_and_ends_on_a_repeat() -> None:
    order = probe.interleave(3)
    assert order == [
        ("kept", 0),
        ("made", 0),
        ("kept", 1),
        ("made", 1),
        ("kept", 2),
        ("kept", 2),
    ]
    # twice the frames, so the clip lasts exactly as long at twice the rate
    assert len(probe.interleave(600)) == 1200
    assert probe.interleave(1) == [("kept", 0), ("kept", 0)]
