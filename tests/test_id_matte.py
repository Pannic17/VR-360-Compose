"""ID mattes (ROADMAP P17, upstream Part 10): a stem the manifest tags as an ID matte is
stitched by the options its picture would be, as a PNG sequence at the picture's size with
a ProRes 4444 copy beside it.

The captures are 15 directories of 64 px tiles, so the native master is 256 x 128. The
delivery sizes are shrunk in the tests that need a resize, so they stay small.
"""

from __future__ import annotations

import json
import pathlib
import struct
import subprocess

import numpy as np
import pytest

from conftest import (
    analytic_panorama,
    make_source_tree,
    manifest_document,
    sample_panorama_into_tile,
)
from vr_compose import encode, manifest
from vr_compose.cli import main
from vr_compose.rig import fifteen_file_rig

STEM = "S.PanoramicIdMatte1"
FRAMES = (1, 2)
TILE, NATIVE = 64, 256


def _matte_capture(root: pathlib.Path, *, tagged: bool = True) -> pathlib.Path:
    """A finished 15-view capture whose one stem the manifest tags as an ID matte, or, not
    `tagged`, a capture without ID groups whose stem is a picture."""
    rig = fifteen_file_rig()
    panorama = analytic_panorama(NATIVE, NATIVE // 2)
    make_source_tree(
        root, cameras=15, stem=STEM, frames=FRAMES, inner=None, size=(TILE, TILE),
        tile_for=lambda camera, _frame: sample_panorama_into_tile(panorama, rig, camera, TILE),
        name_digits=2,
    )  # fmt: skip
    document = manifest_document(stem=STEM, frames=FRAMES, size=TILE)
    document["id_mattes"] = {
        "values": "coverage",
        "groups": [],
        "images": [
            {
                "name": "IdMatte1",
                "pass": "PanoramicIdMatte1",
                "stems": [STEM],
                "files": 30,
                "channels": {
                    "R": {"id": 1, "group": "Creatures"},
                    "G": {"id": 2, "group": "Wreck"},
                    "B": {"id": 3, "group": "Coral"},
                },
            }
        ],
    }
    if not tagged:
        del document["id_mattes"]
    (root / manifest.MANIFEST_NAME).write_text(json.dumps(document), encoding="utf-8")
    return root


def _probe(movie: pathlib.Path) -> dict[str, str]:
    """The first video stream of `movie` as ffprobe reports it."""
    ffprobe = encode._locate("ffprobe")
    assert ffprobe is not None
    shown = subprocess.run(
        [str(ffprobe), "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
         "stream=codec_name,profile,width,height,pix_fmt,r_frame_rate,nb_read_frames",
         "-of", "json", str(movie)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    stream: dict[str, str] = {
        key: str(value) for key, value in json.loads(shown.stdout)["streams"][0].items()
    }
    return stream


def _png_header(path: pathlib.Path) -> tuple[int, int, int, int]:
    """Width, height, bit depth and colour type from a PNG's IHDR."""
    width, height, depth, colour = struct.unpack(">IIBB", path.read_bytes()[16:26])
    return width, height, depth, colour


def _run(tmp_path: pathlib.Path, *extra: str) -> tuple[int, list[pathlib.Path]]:
    source = _matte_capture(tmp_path / "src")
    out = tmp_path / "matte.mp4"
    code = main(["--source", str(source), "sequence", "--out", str(out), "--no-bar", *extra])
    return code, sorted((tmp_path / "matte").glob("*.png"))


def test_a_matte_is_a_png_sequence_resized_to_the_picture_s_size(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Native density above the delivery size: stitched there, resized like the picture."""
    monkeypatch.setitem(encode.SIZES, "4k", (128, 64))
    code, pngs = _run(tmp_path, "--size", "4k", "--codec", "h265", "--fps", "60")
    assert code == 0
    assert [p.name for p in pngs] == [f"{STEM}_S_0001.png", f"{STEM}_S_0002.png"]
    assert {_png_header(p) for p in pngs} == {(128, 64, 16, 2)}, "16-bit RGB by default"
    printed = capsys.readouterr().out
    assert "id matte   : IdMatte1 -- R 1 Creatures, G 2 Wreck, B 3 Coral" in printed
    assert "id matte   : not harmonised, overlap gate off" in printed
    assert "not used   : --codec --" in printed, "the ProRes copy plays at --fps"
    assert "matte      : 128x64 16-bit PNG stitched at 256x128 and resized to the 4k" in printed
    stream = _probe(tmp_path / "matte.mov")
    assert (stream["codec_name"], stream["profile"]) == ("prores", "4444")
    assert stream["pix_fmt"].startswith("yuv444p"), "4:4:4, no chroma decimation"
    assert (stream["width"], stream["height"], stream["nb_read_frames"]) == ("128", "64", "2")
    assert stream["r_frame_rate"] == "60/1"
    assert "matte mov  : 2 frame(s) ->" in printed


def test_no_matte_prores_writes_only_the_pngs(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(encode.SIZES, "4k", (128, 64))
    code, pngs = _run(tmp_path, "--size", "4k", "--no-matte-prores")
    assert code == 0 and len(pngs) == 2
    assert not (tmp_path / "matte.mov").exists()
    assert "matte mov" not in capsys.readouterr().out


def test_no_matte_prores_is_refused_for_a_picture(tmp_path: pathlib.Path) -> None:
    source = _matte_capture(tmp_path / "src", tagged=False)
    with pytest.raises(SystemExit) as caught:
        main(["--source", str(source), "sequence", "--no-matte-prores", "--no-bar"])
    assert "only applies to an ID matte stem" in str(caught.value)


def test_a_matte_below_the_delivery_size_is_stitched_at_it(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Native density below the delivery size: stitched straight at it, as the picture is."""
    monkeypatch.setitem(encode.SIZES, "4k", (512, 256))
    code, pngs = _run(tmp_path, "--size", "4k", "--bit-depth", "8")
    assert code == 0
    assert {_png_header(p) for p in pngs} == {(512, 256, 8, 2)}
    printed = capsys.readouterr().out
    assert "matte      : 512x256 8-bit PNG stitched at the 4k delivery size" in printed
    assert "not used" not in printed


def test_a_matte_with_png_out_is_the_native_master(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _matte_capture(tmp_path / "src")
    out = tmp_path / "masters"
    code = main(
        ["--source", str(source), "sequence", "--out-format", "png", "--size", "4k",
         "--out", str(out), "--no-bar"]
    )  # fmt: skip
    assert code == 0
    assert {_png_header(p) for p in out.glob("*.png")} == {(NATIVE, NATIVE // 2, 16, 2)}
    printed = capsys.readouterr().out
    assert "at the input's own density" in printed
    assert "not used   : --size" in printed, "a master has no delivery size to follow"
    assert not list(tmp_path.glob("*.mov")), "a native master has no ProRes copy"


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_the_resize_keeps_the_depth_and_a_flat_image_flat(dtype: type[np.generic]) -> None:
    value = 40000 if dtype is np.uint16 else 200
    image = np.full((32, 64, 3), value, dtype)
    resized = encode.resize_lanczos(image, 32, 16)
    assert resized.shape == (16, 32, 3) and resized.dtype == dtype
    assert int(np.abs(resized.astype(np.int64) - value).max()) <= 1
