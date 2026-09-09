"""CLI wiring. The GUI in P5 calls these same library functions, so this covers both."""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import pytest

from conftest import analytic_panorama, make_source_tree, sample_panorama_into_tile
from vr_compose import __version__, encode
from vr_compose.cli import _master_size, build_parser, main
from vr_compose.rig import twenty_file_rig


def encode_tools() -> encode.Tools | None:
    try:
        return encode.find_tools()
    except encode.FfmpegNotFound:
        return None


def _tiny_mp4(path: pathlib.Path) -> pathlib.Path:
    """A small delivery file, written by the encoder path the pipeline itself uses."""
    tools = encode_tools()
    assert tools is not None
    spec = encode.EncodeSpec(64, 32, "h264", 400, 30)
    with encode.SegmentWriter(tools, spec, path) as writer:
        for i in range(4):
            writer.write(np.full((32, 64, 3), 40 + 20 * i, np.uint8))
    return path


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


def test_master_size_follows_the_tiles_and_the_delivery_size(tmp_path: pathlib.Path) -> None:
    """The master follows the render, the delivery follows the spec.

    1920 tiles at 8K is the degenerate case -- native density *is* the delivery size, so
    nothing is resampled and the output stays byte-for-byte what P2 verified. A 16K render
    (3840 tiles) delivered at 8K, and any 4K delivery, do get a master.
    """
    from vr_compose import source as source_mod

    make_source_tree(tmp_path, cameras=20, stem="S", frames=[1])
    scanned = next(s for s in source_mod.scan(tmp_path) if s.usable)
    rig = twenty_file_rig()

    def with_tiles(size: int) -> source_mod.SourceSet:
        # the rule depends only on the tile edge, so fake it rather than write 220 MB
        return dataclasses.replace(
            scanned,
            cameras=tuple(dataclasses.replace(c, width=size, height=size) for c in scanned.cameras),
        )

    eight_k = with_tiles(1920)

    assert _master_size(eight_k, rig, "8k", "native") is None, "native density is 8K already"
    assert _master_size(eight_k, rig, "4k", "native") == (7680, 3840), "4K comes off the master"
    assert _master_size(eight_k, rig, "4k", "delivery") is None, "the opt-out preview path"

    sixteen_k = with_tiles(3840)
    assert (eight_k.tile_size, sixteen_k.tile_size) == (1920, 3840)
    assert _master_size(sixteen_k, rig, "8k", "native") == (15360, 7680), "16K render, 8K delivery"
    assert _master_size(sixteen_k, rig, "4k", "native") == (15360, 7680)


def test_stitch_at_defaults_to_native() -> None:
    parser = build_parser()
    assert parser.parse_args(["sequence"]).stitch_at == "native"
    assert parser.parse_args(["sequence", "--stitch-at", "delivery"]).stitch_at == "delivery"


def test_a_cancelled_run_exits_130_not_as_a_failure(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancelled subclasses RuntimeError, so the order of the CLI's except branches is
    load-bearing: caught one branch lower it would become SystemExit(1), i.e. a crash."""
    from vr_compose import pipeline

    rig = twenty_file_rig()
    panorama = analytic_panorama(64, 32)
    make_source_tree(
        tmp_path,
        cameras=20,
        stem="S",
        frames=[1, 2],
        size=(16, 16),
        tile_for=lambda camera, frame: sample_panorama_into_tile(panorama, rig, camera, 16),
    )

    def cancel(*_args: object, **_kwargs: object) -> None:
        raise pipeline.Cancelled("cancelled after 7 of 99 frame(s)")

    monkeypatch.setattr(pipeline, "run_sequence", cancel)
    code = main(["--source", str(tmp_path), "sequence", "--frames", "1-2", "--no-bar"])
    assert code == 130
    err = capsys.readouterr().err
    assert "interrupted" in err and "resume" in err
    assert "cancelled after 7 of 99" in err, "say how far it got"


def _png_source(root: pathlib.Path) -> None:
    rig = twenty_file_rig()
    panorama = analytic_panorama(256, 128)
    make_source_tree(
        root,
        cameras=20,
        stem="S",
        frames=[1, 2],
        size=(64, 64),
        tile_for=lambda camera, _frame: sample_panorama_into_tile(panorama, rig, camera, 64),
    )


def test_png_masters_land_at_the_native_density(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--out` is a directory here, and the width follows the tiles, not `--size`."""
    _png_source(tmp_path / "src")
    out = tmp_path / "masters"
    code = main(
        ["--source", str(tmp_path / "src"), "sequence", "--out-format", "png",
         "--out", str(out), "--no-bar"]
    )  # fmt: skip
    assert code == 0
    assert sorted(p.name for p in out.glob("*.png")) == ["S_S_0001.png", "S_S_0002.png"]
    printed = capsys.readouterr().out
    assert "master     : 256x128 8-bit PNG" in printed, printed
    assert "2 master(s)" in printed


def test_delivery_options_are_refused_for_png_masters(tmp_path: pathlib.Path) -> None:
    """Refuse rather than ignore (AGENTS.md section 10): silently dropping --codec would
    let someone believe they had chosen something."""
    _png_source(tmp_path / "src")
    base = ["--source", str(tmp_path / "src"), "sequence", "--out-format", "png", "--no-bar"]
    for extra in (["--codec", "h265"], ["--size", "4k"], ["--fps", "60"], ["--bitrate", "low"]):
        with pytest.raises(SystemExit) as caught:
            main([*base, *extra])
        assert "only apply to --out-format mp4" in str(caught.value), extra
    # passing a default value explicitly is not "asking for" anything
    assert main([*base, "--codec", "h264", "--out", str(tmp_path / "m")]) == 0


def test_master_options_are_refused_for_mp4(tmp_path: pathlib.Path) -> None:
    _png_source(tmp_path / "src")
    with pytest.raises(SystemExit) as caught:
        main(
            ["--source", str(tmp_path / "src"), "sequence", "--compress-level", "9", "--no-bar"]
        )  # fmt: skip
    assert "only apply to --out-format png" in str(caught.value)


def test_exr_says_why_it_is_not_implemented(tmp_path: pathlib.Path) -> None:
    """The interface is reserved; the reason it is empty is a source-side fact."""
    _png_source(tmp_path / "src")
    with pytest.raises(SystemExit) as caught:
        main(["--source", str(tmp_path / "src"), "sequence", "--out-format", "exr", "--no-bar"])
    message = str(caught.value)
    assert "8-bit" in message and "--out-format png" in message


def test_auto_named_output_uses_the_stem_and_a_timestamp(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """P5 naming, both modes, and the stamp taken exactly once per run.

    Once matters: for the MP4 path the segment directory is derived from the output name,
    so a second reading of the clock could scatter one job's segments across two places.
    """
    from vr_compose import pipeline

    _png_source(tmp_path / "src")
    calls: list[str] = []

    def one_stamp(*_args: object, **_kwargs: object) -> str:
        calls.append("read")
        return f"0908_163{len(calls)}"

    monkeypatch.setattr(pipeline, "run_stamp", one_stamp)
    monkeypatch.setattr(pipeline, "default_output_dir", lambda: tmp_path / "beside")

    assert main(["--source", str(tmp_path / "src"), "sequence", "--out-format", "png",
                 "--no-bar"]) == 0  # fmt: skip
    assert calls == ["read"], "the clock is read once per run"
    made = sorted(p.name for p in (tmp_path / "beside").iterdir())
    assert made == ["S_0908_1631"], made
    assert sorted(p.name for p in (tmp_path / "beside" / made[0]).glob("*.png")) == [
        "S_S_0001.png",
        "S_S_0002.png",
    ]

    calls.clear()
    printed = capsys.readouterr()
    assert "S_0908_1631" in printed.out, "the chosen path is printed, to allow resume"


def test_two_auto_named_runs_in_one_minute_do_not_share_a_directory(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`MMDD_HHMM` can repeat, so the second run gets `_2` rather than the first's output.

    P5 promises an auto-generated name is a fresh output every run -- that is why it does
    not resume. With seconds in the stamp the promise held by itself; with the width the
    user asked for on 2026-09-08 it needs `pipeline.unique_path`, and this is the test
    that says so. Without it the second run would adopt the first run's finished frames,
    which is wrong the moment any setting differs between them.
    """
    from vr_compose import pipeline

    _png_source(tmp_path / "src")
    monkeypatch.setattr(pipeline, "run_stamp", lambda *_a, **_k: "0908_1630")
    monkeypatch.setattr(pipeline, "default_output_dir", lambda: tmp_path / "beside")

    argv = ["--source", str(tmp_path / "src"), "sequence", "--out-format", "png", "--no-bar"]
    assert main(argv) == 0
    assert main(argv) == 0

    made = sorted(p.name for p in (tmp_path / "beside").iterdir())
    assert made == ["S_0908_1630", "S_0908_1630_2"], made
    for name in made:
        assert sorted(p.name for p in (tmp_path / "beside" / name).glob("*.png")) == [
            "S_S_0001.png",
            "S_S_0002.png",
        ], f"{name} is a complete run of its own, not half of a shared one"


def test_the_video_name_is_stem_and_timestamp(tmp_path: pathlib.Path) -> None:
    from vr_compose import pipeline
    from vr_compose import source as source_mod

    _png_source(tmp_path / "src")
    chosen = next(s for s in source_mod.scan(tmp_path / "src") if s.usable)
    assert pipeline.default_output_name(chosen, "0908_1630") == "S_0908_1630.mp4"


# --- 360 metadata on an existing file ------------------------------------------------


def test_metadata_reports_absence_then_writes_it(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`vr-compose metadata` is for files rendered before the metadata existed."""
    if encode_tools() is None:
        pytest.skip("ffmpeg not available")
    path = _tiny_mp4(tmp_path / "old.mp4")
    assert main(["metadata", str(path)]) == 1, "no metadata is a failure exit, so scripts see it"
    assert "absent" in capsys.readouterr().out

    assert main(["metadata", str(path), "--write"]) == 0
    out = capsys.readouterr().out
    assert "wrote" in out and "equirectangular" in out
    # our parser and ffprobe both, because the command cross-checks them
    assert out.count("equirectangular") == 2

    assert main(["metadata", str(path), "--write"]) == 0
    assert "already carries" in capsys.readouterr().out


def test_metadata_refuses_a_missing_file(tmp_path: pathlib.Path) -> None:
    with pytest.raises(SystemExit, match="no such file"):
        main(["metadata", str(tmp_path / "nope.mp4")])


def test_metadata_says_what_is_wrong_with_a_file_it_cannot_annotate(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "junk.mp4"
    path.write_bytes(b"not an mp4 at all, but long enough to walk")
    before = path.read_bytes()
    # The complaint names the box that did not add up, rather than a traceback -- and the
    # file is left exactly as it was, which is the part that matters for a delivery.
    with pytest.raises(SystemExit, match=r"claims .* bytes, past the file"):
        main(["metadata", str(path), "--write"])
    assert path.read_bytes() == before


# --- --device -----------------------------------------------------------------------------


def test_device_is_a_cli_option_defaulting_to_cpu() -> None:
    """GPU acceleration is opt-in from the command line only (user, 2026-09-09)."""
    parser = build_parser()
    for sub in ("frame", "sequence"):
        args = parser.parse_args([sub])
        assert args.device == "cpu", sub
        assert parser.parse_args([sub, "--device", "cuda"]).device == "cuda"
        assert parser.parse_args([sub, "--device", "auto"]).device == "auto"
    with pytest.raises(SystemExit):
        parser.parse_args(["sequence", "--device", "opencl"])


def test_frame_with_cuda_but_no_gpu_warns_and_still_stitches(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from vr_compose import device

    monkeypatch.setattr(device, "probe_cuda", lambda: device.CudaProbe(importable=False, error="x"))
    rig = twenty_file_rig()
    panorama = analytic_panorama(512, 256)
    tiles = {i: sample_panorama_into_tile(panorama, rig, i, 128) for i in range(1, 21)}
    make_source_tree(
        tmp_path, cameras=20, stem="Scene_A", frames=[5], size=(128, 128),
        tile_for=lambda index, _frame: tiles[index],
    )  # fmt: skip
    out = tmp_path / "out" / "pano.png"
    argv = ["--source", str(tmp_path), "frame", "--frame", "5", "--width", "512",
            "--out", str(out), "--device", "cuda"]  # fmt: skip
    assert main(argv) == 0
    text = capsys.readouterr().out
    assert "WARNING    : GPU requested" in text and "falling back to the CPU" in text
    assert "PASS" in text and out.exists()


def _gpu_usable() -> bool:
    from vr_compose import device

    found = device.probe_cuda()
    return found.importable and not found.error and found.count > 0


@pytest.mark.skipif(not _gpu_usable(), reason="no usable CUDA device")
def test_frame_on_cuda_writes_the_cpu_bytes(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from vr_compose import device

    monkeypatch.setattr(device, "MIN_CUDA_MEMORY_BYTES", 1)
    rig = twenty_file_rig()
    panorama = analytic_panorama(512, 256)
    tiles = {i: sample_panorama_into_tile(panorama, rig, i, 128) for i in range(1, 21)}
    make_source_tree(
        tmp_path, cameras=20, stem="Scene_A", frames=[5], size=(128, 128),
        tile_for=lambda index, _frame: tiles[index],
    )  # fmt: skip
    common = ["--source", str(tmp_path), "frame", "--frame", "5", "--width", "512", "--out"]
    assert main([*common, str(tmp_path / "cuda.png"), "--device", "cuda"]) == 0
    assert "warp       : cuda" in capsys.readouterr().out
    assert main([*common, str(tmp_path / "cpu.png")]) == 0
    assert (tmp_path / "cuda.png").read_bytes() == (tmp_path / "cpu.png").read_bytes()
