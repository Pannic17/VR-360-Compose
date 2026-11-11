"""The capture's manifest (upstream H1, ROADMAP P11): when it decides the rig, and why not.

Synthetic captures are 15 directories `Camera01..15` with a manifest shaped like the ones
VROpPanoramic writes. The flat ones are deliberate: the layout gate cannot check them
(UNVERIFIABLE), so a run that gets through has had its rig from the manifest. The three
real captures of 2026-11-03 are read too, when they are on this machine.
"""

from __future__ import annotations

import fractions
import json
import pathlib
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from conftest import (
    make_source_tree,
    manifest_document,
    sample_panorama_into_tile,
    textured_panorama,
)
from vr_compose import layout, manifest, pipeline, source
from vr_compose.cli import _delivery_fps, main
from vr_compose.rig import REGISTRY, UnknownRigError, View, fifteen_file_rig

U8 = npt.NDArray[np.uint8]
Document = dict[str, Any]
STEM = "Scene"
FRAMES = (600, 601)
TILE = 16

H1_CAPTURES = pathlib.Path("E:/VR-DEV/Render/VROpPanoramic/H1")
"""The captures upstream wrote when it landed the manifest, when present."""


def _document(
    views: list[View] | None = None, *, size: int = TILE, rate: tuple[int, int] = (30, 1)
) -> Document:
    """A finished manifest describing `_capture`'s files."""
    return manifest_document(views, stem=STEM, frames=FRAMES, size=size, rate=rate)


def _flat(_index: int, _frame: int) -> U8:
    """Every camera the same grey: the overlaps agree and the layout gate cannot check."""
    return np.full((TILE, TILE, 3), 128, np.uint8)


def _capture(
    root: pathlib.Path,
    document: Document | None,
    *,
    cameras: int = 15,
    size: int = TILE,
    tile_for: Callable[[int, int], U8] = _flat,
) -> source.SourceSet:
    make_source_tree(
        root, cameras=cameras, stem=STEM, frames=FRAMES, inner=None, size=(size, size),
        tile_for=tile_for, name_digits=2,
    )  # fmt: skip
    if document is not None:
        (root / manifest.MANIFEST_NAME).write_text(json.dumps(document), encoding="utf-8")
    return source.scan(root)[0]


# -- the manifest decides -------------------------------------------------------------------


def test_a_finished_manifest_builds_the_registered_rig(tmp_path: pathlib.Path) -> None:
    choice = manifest.choose_rig(_capture(tmp_path, _document()))
    assert choice.manifest is not None and choice.set_aside is None
    assert choice.rig.name == "pano-15"
    assert choice.rig.views == fifteen_file_rig().views, "the proved geometry, exactly"
    assert not choice.rig.needs_layout_check
    assert choice.rate == 30
    assert choice.lines() == [
        "manifest   : vrop_panoramic_manifest.json -- schema 1, finished, SectorGrid, "
        "15 views relative to the world, 30 fps",
        "layout     : from manifest -- the layout gate is not run",
    ]


def test_without_a_manifest_the_registry_decides(tmp_path: pathlib.Path) -> None:
    choice = manifest.choose_rig(_capture(tmp_path, None))
    assert choice.rig is REGISTRY[15] and choice.rig.needs_layout_check
    assert choice.manifest is None and choice.rate is None
    assert choice.lines() == ["manifest   : none"]


def test_the_directory_not_the_entry_order_names_the_camera(tmp_path: pathlib.Path) -> None:
    document = _document()
    document["rig"]["views"].reverse()
    choice = manifest.choose_rig(_capture(tmp_path, document))
    assert choice.rig.views == fifteen_file_rig().views


def test_a_reordered_rig_is_stitched_in_the_manifest_s_order(tmp_path: pathlib.Path) -> None:
    """The manifest, not the numbering, says where each directory looked."""
    views = list(fifteen_file_rig().views)
    views[0], views[3] = views[3], views[0]
    choice = manifest.choose_rig(_capture(tmp_path, _document(views)))
    assert choice.rig.name == "pano-15"
    assert choice.rig.views[0] == View(72.0, 45.0) and choice.rig.views[3] == View(0.0, 45.0)
    assert choice.rig.views[1:3] == fifteen_file_rig().views[1:3]


def test_several_stems_bound_the_frames_rather_than_count_them(tmp_path: pathlib.Path) -> None:
    document = _document()
    document["tiles"]["stems"] = [STEM, "Other"]
    for view in document["rig"]["views"]:
        view["written"] = {"count": 9, "first": 599, "last": 610}
    assert manifest.choose_rig(_capture(tmp_path / "a", document)).manifest is not None
    for view in document["rig"]["views"]:
        view["written"]["first"] = 601
    choice = manifest.choose_rig(_capture(tmp_path / "b", document))
    assert choice.set_aside is not None and "Camera01: it records 9 frame(s)" in choice.set_aside


# -- the manifest is set aside, and today's path runs ----------------------------------------


def _set(*path: str | int, value: object) -> Callable[[Document], None]:
    def edit(document: Document) -> None:
        target: Any = document
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return edit


def _delete(*path: str | int) -> Callable[[Document], None]:
    def edit(document: Document) -> None:
        target: Any = document
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]

    return edit


def _rendering(document: Document) -> None:
    document["state"] = "rendering"
    document["tiles"]["stems"] = []
    for view in document["rig"]["views"]:
        del view["written"]


SET_ASIDE: list[tuple[str, Callable[[Document], None], str]] = [
    ("schema", _set("schema", value="other"), "schema 'other' is not"),
    ("version", _set("schema_version", value=2), "schema version 2; this VR-Compose reads 1"),
    ("rendering", _rendering, "state 'rendering': the capture did not finish"),
    ("failed", _set("state", value="failed"), "state 'failed': Movie Render Queue"),
    ("other stem", _set("tiles", "stems", value=["Other"]), "lists stems 'Other', not 'Scene'"),
    ("field", _delete("rig", "views", 3, "pitch"), "rig.views[3].pitch is missing"),
    ("type", _set("rig", "views", 3, "yaw", value="72"), "rig.views[3].yaw is not a number"),
    ("axes", _set("rig", "coordinate_system", value="z_up"), "rig.coordinate_system 'z_up'"),
    ("reference", _set("rig", "orientation_reference", value="hmd"), "orientation_reference"),
    ("format", _set("tiles", "format", value="exr"), "exr tiles; the disk holds png"),
    ("encoding", _set("tiles", "encoding", value="linear_half"), "encoded as linear_half"),
    ("size", _set("tiles", "width", value=32), "32x16 tiles; the disk holds 16x16"),
    (
        "directory",
        _set("rig", "views", 14, "directory", value="Camera16"),
        "Camera16 in it but not on disk; Camera15 on disk but not in it",
    ),
    ("twice", _set("rig", "views", 14, "directory", value="Camera01"), "Camera01 more than once"),
    (
        "written",
        _set("rig", "views", 2, "written", "count", value=3),
        "Camera03: it records 3 frame(s) 600..601 written, the disk holds 2 frame(s) 600..601",
    ),
    ("unwritten", _delete("rig", "views", 0, "written"), "no frames written for Camera01"),
    ("rate", _set("frames", "rate", "numerator", value=0), "0/1 is not a positive rate"),
]


@pytest.mark.parametrize(
    ("edit", "reason"), [case[1:] for case in SET_ASIDE], ids=[case[0] for case in SET_ASIDE]
)
def test_an_unusable_manifest_is_set_aside_with_its_reason(
    tmp_path: pathlib.Path, edit: Callable[[Document], None], reason: str
) -> None:
    document = _document()
    edit(document)
    choice = manifest.choose_rig(_capture(tmp_path, document))
    assert choice.manifest is None and choice.rate is None
    assert choice.rig is REGISTRY[15] and choice.rig.needs_layout_check, "the gate still runs"
    assert choice.set_aside is not None and reason in choice.set_aside, choice.set_aside
    assert choice.lines() == [f"manifest   : not used -- {choice.set_aside}"]


@pytest.mark.parametrize(
    ("text", "reason"),
    [("{", "is not JSON"), ("[]", "the file is not a JSON object"), ("", "is not JSON")],
)
def test_a_file_that_is_not_a_manifest_is_set_aside(
    tmp_path: pathlib.Path, text: str, reason: str
) -> None:
    found = _capture(tmp_path, None)
    (tmp_path / manifest.MANIFEST_NAME).write_text(text, encoding="utf-8")
    choice = manifest.choose_rig(found)
    assert choice.set_aside is not None and reason in choice.set_aside
    assert choice.rig is REGISTRY[15]


def test_a_set_aside_manifest_is_named_when_the_registry_refuses(tmp_path: pathlib.Path) -> None:
    """A directory gone missing: no rig either way, and the refusal says why the manifest
    did not decide it."""
    with pytest.raises(UnknownRigError, match="no rig registered for 14 cameras") as caught:
        manifest.choose_rig(_capture(tmp_path, _document(), cameras=14))
    assert "The manifest was not used: its directories are not the disk's: Camera15 in it" in (
        str(caught.value)
    )


# -- a usable manifest describing a rig nobody registered ------------------------------------


def _cubemap() -> list[View]:
    return [View(0, 0), View(90, 0), View(180, 0), View(-90, 0), View(0, 90), View(0, -90)]


def _rolled(document: Document) -> None:
    document["rig"]["views"][4]["roll"] = 5


def _wider(document: Document) -> None:
    for view in document["rig"]["views"]:
        view["fov"] = 100


def _mixed_fov(document: Document) -> None:
    document["rig"]["views"][7]["fov"] = 91


def _tilted(document: Document) -> None:
    document["rig"]["views"][14]["pitch"] = 30


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (_rolled, r"rolled views \(Camera05\)"),
        (_wider, "15 views at 100 deg match no registered rig"),
        (_mixed_fov, "different fields of view"),
        (_tilted, "15 views at 90 deg match no registered rig"),
    ],
    ids=["rolled", "wider", "mixed-fov", "tilted"],
)
def test_a_rig_the_registry_does_not_have_is_refused(
    tmp_path: pathlib.Path, edit: Callable[[Document], None], message: str
) -> None:
    document = _document()
    edit(document)
    with pytest.raises(UnknownRigError, match=message):
        manifest.choose_rig(_capture(tmp_path, document))


def test_a_six_view_cubemap_is_refused_until_a_six_view_rig_exists(
    tmp_path: pathlib.Path,
) -> None:
    document = _document(_cubemap())
    document["rig"]["type"] = "Cubemap"
    with pytest.raises(UnknownRigError, match="the manifest's Cubemap rig: 6 views at 90 deg"):
        manifest.choose_rig(_capture(tmp_path, document, cameras=6))


# -- what a run reports, and the layout gate -----------------------------------------------


def test_frame_takes_the_rig_from_the_manifest_and_skips_the_gate(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the layout gate ran although the manifest decided the rig")

    _capture(tmp_path / "src", _document())
    monkeypatch.setattr(layout, "check", refuse)
    out = tmp_path / "pano.png"
    assert main(["--source", str(tmp_path / "src"), "frame", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "rig        : pano-15, reading 15 of 15 files" in text
    assert "manifest   : vrop_panoramic_manifest.json -- schema 1, finished" in text
    assert "layout     : from manifest -- the layout gate is not run" in text
    assert "NOT VERIFIED" not in text
    assert out.exists()


@pytest.mark.parametrize("state", [None, "rendering"], ids=["no-manifest", "rendering"])
def test_without_a_usable_manifest_the_layout_gate_decides(
    tmp_path: pathlib.Path, state: str | None, capsys: pytest.CaptureFixture[str]
) -> None:
    """The same flat capture: what the manifest let through, the gate cannot check."""
    document = None
    if state is not None:
        document = _document()
        _rendering(document)
    _capture(tmp_path / "src", document)
    out = tmp_path / "pano.png"
    with pytest.raises(SystemExit, match="layout gate UNVERIFIABLE"):
        main(["--source", str(tmp_path / "src"), "frame", "--out", str(out)])
    expected = "none" if state is None else "not used -- state 'rendering'"
    assert f"manifest   : {expected}" in capsys.readouterr().out
    assert not out.exists()


def test_the_pipeline_skips_the_gate_for_a_rig_from_a_manifest(tmp_path: pathlib.Path) -> None:
    found = _capture(tmp_path, _document())
    rig = manifest.choose_rig(found).rig
    assert pipeline.check_layout(found, rig, FRAMES[0], "on", print) is None
    with pytest.raises(layout.LayoutGateFailed, match="UNVERIFIABLE"):
        pipeline.check_layout(found, REGISTRY[15], FRAMES[0], "on", lambda _: None)


def test_discover_reports_the_manifest(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _capture(tmp_path / "src", _document())
    assert main(["--source", str(tmp_path / "src"), "discover"]) == 0
    text = capsys.readouterr().out
    assert "layout     : from manifest" in text
    assert "the layout gate checks them" not in text


@pytest.fixture(scope="module")
def reordered() -> dict[int, U8]:
    """512 px tiles of a textured panorama, with Camera01 and Camera04 swapped on disk."""
    rig = fifteen_file_rig()
    panorama = textured_panorama(2048, 1024)
    tiles = {k: sample_panorama_into_tile(panorama, rig, k, 512) for k in range(1, 16)}
    tiles[1], tiles[4] = tiles[4], tiles[1]
    return tiles


def test_a_capture_numbered_otherwise_is_stitched_by_its_manifest(
    tmp_path: pathlib.Path, reordered: dict[int, U8], capsys: pytest.CaptureFixture[str]
) -> None:
    """End to end: the numbering alone is refused, the manifest's word stitches it right."""
    views = list(fifteen_file_rig().views)
    views[0], views[3] = views[3], views[0]

    def tile(index: int, _frame: int) -> U8:
        return reordered[index]

    _capture(tmp_path / "bare", None, size=512, tile_for=tile)
    with pytest.raises(SystemExit, match="layout gate"):
        main(["--source", str(tmp_path / "bare"), "frame", "--out", str(tmp_path / "a.png")])

    _capture(tmp_path / "told", _document(views, size=512), size=512, tile_for=tile)
    code = main(["--source", str(tmp_path / "told"), "frame", "--out", str(tmp_path / "b.png")])
    text = capsys.readouterr().out
    assert code == 0, text
    assert "verdict    : PASS" in text


# -- the delivery frame rate -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("requested", "rate", "fps", "line"),
    [
        (None, None, 30, None),
        (60, None, 60, None),
        (None, 60, 60, "fps        : 60 from the manifest"),
        (None, 30, 30, "fps        : 30 from the manifest"),
        (60, 60, 60, None),
        (
            30,
            60,
            30,
            "WARNING    : --fps 30, but the capture is 60 fps (manifest frames.rate); "
            "the video plays at 0.50x its speed",
        ),
        (30, 24, 30, "WARNING    : --fps 30, but the capture is 24 fps"),
    ],
)
def test_the_delivery_rate_follows_the_manifest_unless_asked(
    requested: int | None, rate: int | None, fps: int, line: str | None
) -> None:
    capture = None if rate is None else fractions.Fraction(rate)
    chosen, said = _delivery_fps(requested, capture)
    assert chosen == fps
    if line is None:
        assert said is None
    else:
        assert said is not None and said.startswith(line)


@pytest.mark.parametrize(
    ("rate", "shown"),
    [(fractions.Fraction(24), "24 fps"), (fractions.Fraction(30000, 1001), "29.97 fps")],
)
def test_a_capture_rate_no_delivery_has_must_be_chosen(
    rate: fractions.Fraction, shown: str
) -> None:
    with pytest.raises(SystemExit, match="Pass --fps to choose one") as caught:
        _delivery_fps(None, rate)
    assert shown in str(caught.value)


def test_sequence_encodes_at_the_capture_s_rate(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[pipeline.SequenceJob] = []

    def capture(job: pipeline.SequenceJob, **_kwargs: object) -> None:
        seen.append(job)
        raise pipeline.Cancelled("stopped by the test")

    monkeypatch.setattr(pipeline, "run_sequence", capture)
    _capture(tmp_path / "src", _document(rate=(60, 1)))
    base = ["--source", str(tmp_path / "src"), "sequence", "--no-bar"]
    assert main(base) == 130
    assert "fps        : 60 from the manifest" in capsys.readouterr().out
    assert main([*base, "--fps", "30"]) == 130
    assert "WARNING    : --fps 30, but the capture is 60 fps" in capsys.readouterr().out
    assert [job.spec.fps for job in seen] == [60, 30]


def test_a_master_does_not_ask_for_a_rate(tmp_path: pathlib.Path) -> None:
    """Frames have no rate to choose, so a 24 fps capture's masters are not refused."""
    _capture(tmp_path / "src", _document(rate=(24, 1)))
    base = ["--source", str(tmp_path / "src"), "sequence", "--no-bar"]
    with pytest.raises(SystemExit, match="Pass --fps"):
        main(base)
    out = tmp_path / "masters"
    assert main([*base, "--out-format", "png", "--out", str(out)]) == 0
    with pytest.raises(SystemExit, match="only apply to --out-format mp4"):
        main([*base, "--out-format", "png", "--fps", "30", "--out", str(tmp_path / "m2")])


# -- the real captures ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "rate", "tile_format", "tile"),
    [
        ("20261103_151501_B_Bay_submerged_ships_Demo_02", 30, "png", 2560),
        ("20261103_150926_B_Bay_submerged_ships_Demo_02", 60, "exr", 512),
    ],
)
def test_upstream_s_finished_captures_build_the_registered_rig(
    name: str, rate: int, tile_format: str, tile: int
) -> None:
    root = H1_CAPTURES / name
    if not (root / manifest.MANIFEST_NAME).is_file():
        pytest.skip(f"no upstream capture at {root}")
    (found,) = (s for s in source.scan(root) if s.usable)
    assert (found.tile_format, found.tile_size) == (tile_format, tile)
    choice = manifest.choose_rig(found)
    assert choice.manifest is not None, choice.set_aside
    assert choice.rig.views == fifteen_file_rig().views
    assert not choice.rig.needs_layout_check
    assert choice.rate == rate


def test_upstream_s_killed_capture_falls_back_to_the_gate() -> None:
    root = H1_CAPTURES / "20261103_151244_B_Bay_submerged_ships_Demo_02"
    if not (root / manifest.MANIFEST_NAME).is_file():
        pytest.skip(f"no upstream capture at {root}")
    (found,) = (s for s in source.scan(root) if s.usable)
    choice = manifest.choose_rig(found)
    assert choice.set_aside == "state 'rendering': the capture did not finish"
    assert choice.rig is REGISTRY[15]


# -- ID matte tags (ROADMAP P17, upstream Part 10) ------------------------------------------


def _with_id_mattes(document: Document, stems: list[str]) -> Document:
    """`document` with one ID matte image holding two groups, R and G, on `stems`."""
    document["id_mattes"] = {
        "values": "coverage",
        "groups": [],
        "images": [
            {
                "name": "IdMatte1",
                "pass": "PanoramicIdMatte1",
                "stems": stems,
                "files": 30,
                "channels": {
                    "R": {"id": 1, "group": "Creatures"},
                    "G": {"id": 2, "group": "Wreck"},
                    "B": None,
                },
            }
        ],
    }
    return document


def test_a_stem_the_manifest_tags_as_an_id_matte_is_one(tmp_path: pathlib.Path) -> None:
    choice = manifest.choose_rig(_capture(tmp_path, _with_id_mattes(_document(), [STEM])))
    assert choice.manifest is not None and choice.set_aside is None
    assert choice.id_matte is not None
    assert choice.id_matte.channels == (
        manifest.IdMatteChannel("R", 1, "Creatures"),
        manifest.IdMatteChannel("G", 2, "Wreck"),
    )
    assert choice.lines()[-1] == "id matte   : IdMatte1 -- R 1 Creatures, G 2 Wreck"


def test_the_picture_s_stem_beside_its_mattes_is_not_one(tmp_path: pathlib.Path) -> None:
    document = _with_id_mattes(_document(), [f"{STEM}.PanoramicIdMatte1"])
    choice = manifest.choose_rig(_capture(tmp_path, document))
    assert choice.manifest is not None and choice.id_matte is None
    assert len(choice.manifest.id_mattes) == 1
    assert not any(line.startswith("id matte") for line in choice.lines())


def test_a_capture_without_id_groups_has_no_mattes(tmp_path: pathlib.Path) -> None:
    choice = manifest.choose_rig(_capture(tmp_path, _document()))
    assert choice.manifest is not None and choice.manifest.id_mattes == ()
    assert choice.id_matte is None


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (_set("id_mattes", "images", value={}), "id_mattes.images is not a list"),
        (
            _delete("id_mattes", "images", 0, "channels", "R", "id"),
            "id_mattes.images[0].channels.R.id is missing",
        ),
        (
            _set("id_mattes", "images", 0, "stems", value="Scene"),
            "id_mattes.images[0].stems is not a list of strings",
        ),
    ],
)
def test_malformed_id_matte_tags_set_the_manifest_aside(
    tmp_path: pathlib.Path, mutate: Callable[[Document], None], reason: str
) -> None:
    document = _with_id_mattes(_document(), [STEM])
    mutate(document)
    choice = manifest.choose_rig(_capture(tmp_path, document))
    assert choice.manifest is None and choice.set_aside == reason
