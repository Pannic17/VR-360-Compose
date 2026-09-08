"""The P6 window, and the CLI surface it drives.

Two halves, and the split matters. The machine-readable CLI options are tested without
Qt at all, because they are the contract: anything can drive a render by reading NDJSON
and writing a line to cancel. The window is then tested offscreen, and only for the
things a window is responsible for -- the argv it assembles, what it enables, and
whether it refuses to guess when a directory holds several scenes.

There is no test of "does the stitch work through the GUI" because there is nothing
there to test: the job is the CLI, which the rest of this suite covers.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
from collections.abc import Callable

import pytest

from conftest import analytic_panorama, make_source_tree, sample_panorama_into_tile
from vr_compose import pipeline
from vr_compose.rig import twenty_file_rig

pytest.importorskip("PySide6", reason="the GUI needs PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QComboBox

from vr_compose.gui.window import CHOOSE_STEM, ComposeWindow, worker_command

TILE, NATIVE = 64, 256


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    """One QApplication for the module; Qt allows exactly one per process."""
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        return existing
    return QApplication([])


def _source(root: pathlib.Path, stem: str = "S", frames: tuple[int, ...] = (1, 2, 3)) -> None:
    rig = twenty_file_rig()
    panorama = analytic_panorama(NATIVE, NATIVE // 2)
    make_source_tree(
        root,
        cameras=20,
        stem=stem,
        frames=frames,
        size=(TILE, TILE),
        tile_for=lambda camera, _frame: sample_panorama_into_tile(panorama, rig, camera, TILE),
    )


# --- the contract the GUI depends on, tested without Qt --------------------------------


def test_contiguous_tail_skips_the_stray_leading_frame() -> None:
    """The reference set is 0000 plus 1656..2433, and "all" would jump after frame one."""
    assert pipeline.contiguous_tail([0, *range(1656, 2434)]) == (1656, 2433)
    assert pipeline.contiguous_tail([5, 6, 7]) == (5, 7)
    assert pipeline.contiguous_tail([9]) == (9, 9)
    assert pipeline.contiguous_tail([1, 2, 9, 10]) == (9, 10)
    with pytest.raises(ValueError, match="no frames"):
        pipeline.contiguous_tail([])


def test_progress_json_is_ndjson_on_stdout_and_prose_on_stderr(tmp_path: pathlib.Path) -> None:
    """The two streams must stay apart, or a parser chokes on a header line."""
    _source(tmp_path / "src")
    result = subprocess.run(
        [sys.executable, "-m", "vr_compose", "--source", str(tmp_path / "src"), "sequence",
         "--out-format", "png", "--out", str(tmp_path / "out"), "--no-bar", "--progress-json"],
        capture_output=True, text=True, timeout=300,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr

    events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    kinds = [event["event"] for event in events]
    assert kinds[0] == "start" and kinds[-1] == "done"
    assert kinds.count("progress") == 3

    start = events[0]
    assert start["mode"] == "frames" and start["total"] == 3
    assert start["baseline_seconds_per_frame"] == pytest.approx(46.85)

    progress = next(event for event in events if event["event"] == "progress")
    for field in ("done", "total", "frame", "seconds_per_frame", "eta_seconds", "speedup"):
        assert field in progress, field

    assert events[-1]["ok"] is True and events[-1]["problems"] == []
    # the prose went the other way, and none of it landed in the parsed stream
    assert "source     :" in result.stderr
    assert "source     :" not in result.stdout


def test_cancel_on_stdin_stops_cleanly_and_leaves_a_resumable_job(
    tmp_path: pathlib.Path,
) -> None:
    """A line on stdin is the GUI's cancel button; EOF means the parent is gone.

    The guarantee under test is the pipeline's, restated across a process boundary:
    finished frames are kept, no partial file is left, and a re-run finishes the job.
    """
    _source(tmp_path / "src", frames=tuple(range(1, 9)))
    out = tmp_path / "out"
    proc = subprocess.Popen(
        [sys.executable, "-m", "vr_compose", "--source", str(tmp_path / "src"), "sequence",
         "--out-format", "png", "--out", str(out), "--no-bar", "--progress-json",
         "--cancel-on-stdin"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1,
    )  # fmt: skip
    assert proc.stdout is not None and proc.stdin is not None

    kinds: list[str] = []
    asked = False
    for line in proc.stdout:
        event = json.loads(line)
        kinds.append(event["event"])
        if event["event"] == "progress" and not asked:
            proc.stdin.write("cancel\n")
            proc.stdin.flush()
            asked = True
    assert proc.wait(timeout=120) == 130, "cancellation exits 130"
    assert "cancelled" in kinds and "done" not in kinds

    written = sorted(p.name for p in out.glob("*.png"))
    assert written, "the frames finished before the cancel are kept"
    assert len(written) < 8, "and it really did stop early"
    assert not list(out.glob("*.part")), "nothing half-written"

    finished = subprocess.run(
        [sys.executable, "-m", "vr_compose", "--source", str(tmp_path / "src"), "sequence",
         "--out-format", "png", "--out", str(out), "--no-bar"],
        capture_output=True, text=True, timeout=300,
    )  # fmt: skip
    assert finished.returncode == 0, finished.stderr
    assert len(list(out.glob("*.png"))) == 8, "the re-run completed what was left"


# --- the window ------------------------------------------------------------------------


def test_the_window_builds_and_is_styled(qt_app: QApplication, tmp_path: pathlib.Path) -> None:
    window = ComposeWindow()
    try:
        assert window.styleSheet(), "the .qss has to reach the window or nothing is styled"
        assert window.start_button.property("variant") == "primary"
        assert window.cancel_button.property("variant") == "ghost"
        assert not window.cancel_button.isEnabled(), "nothing to cancel yet"
    finally:
        window.close()


def test_a_valid_source_fills_in_the_rig_and_a_safe_frame_range(
    qt_app: QApplication, tmp_path: pathlib.Path
) -> None:
    _source(tmp_path / "src", frames=(0, 5, 6, 7))
    window = ComposeWindow()
    try:
        window.source_edit.setText(str(tmp_path / "src"))
        window.discover()
        assert window.rig_badge.text() == "rig of3d-20"
        assert window.status_banner.property("status") == "pending"
        assert window.start_button.isEnabled()
        # 0 is not contiguous with 5..7, so the prefill is the tail and the log says why
        assert window.frames_edit.text() == "5-7"
        assert "0000" in window.log_edit.toPlainText()
    finally:
        window.close()


def test_a_rejected_directory_shows_the_reason_it_gave(
    qt_app: QApplication, tmp_path: pathlib.Path
) -> None:
    """ROADMAP P6 item 5: the specific problem, not "invalid directory"."""
    make_source_tree(tmp_path / "bad", cameras=4, stem="S", frames=[1], size=(24, 16))
    window = ComposeWindow()
    try:
        window.source_edit.setText(str(tmp_path / "bad"))
        window.discover()
        log = window.log_edit.toPlainText()
        assert "not square" in log, log
        assert window.status_banner.property("status") == "fail"
        assert not window.start_button.isEnabled()
        assert window.build_command() is None
    finally:
        window.close()


def test_several_scenes_are_not_chosen_for_the_user(
    qt_app: QApplication, tmp_path: pathlib.Path
) -> None:
    """ROADMAP P6 item 6. A combo always has a selection, so the selection is a prompt."""
    _source(tmp_path / "src", stem="Scene_A")
    _source(tmp_path / "src", stem="Scene_B")
    window = ComposeWindow()
    try:
        window.source_edit.setText(str(tmp_path / "src"))
        window.discover()
        items = [window.stem_combo.itemText(i) for i in range(window.stem_combo.count())]
        assert items == [CHOOSE_STEM, "Scene_A", "Scene_B"], items
        assert window.stem_combo.currentText() == CHOOSE_STEM
        assert not window.start_button.isEnabled(), "no guessing"
        assert window.build_command() is None

        window.stem_combo.setCurrentText("Scene_B")
        assert window.start_button.isEnabled()
        argv = window.build_command()
        assert argv is not None and "Scene_B" in argv
    finally:
        window.close()


def test_the_argv_carries_only_what_the_chosen_mode_accepts(
    qt_app: QApplication, tmp_path: pathlib.Path
) -> None:
    """The CLI *refuses* inapplicable options, so sending them would break the run."""
    _source(tmp_path / "src")
    window = ComposeWindow()
    try:
        window.source_edit.setText(str(tmp_path / "src"))
        window.discover()
        window.frames_edit.setText("1-3")

        window.video_radio.setChecked(True)
        video = window.build_command()
        assert video is not None
        assert "--size" in video and "--codec" in video and "--fps" in video
        assert "--bit-depth" not in video and "--out-format" not in video
        assert window.bit_depth_combo.isEnabled() is False

        window.frames_radio.setChecked(True)
        frames = window.build_command()
        assert frames is not None
        assert frames[frames.index("--out-format") + 1] == "png"
        assert "--bit-depth" in frames
        for delivery_only in ("--size", "--codec", "--bitrate", "--fps"):
            assert delivery_only not in frames, delivery_only
        assert window.size_combo.isEnabled() is False

        for argv in (video, frames):
            assert argv[: len(worker_command())] == worker_command()
            assert "--progress-json" in argv and "--cancel-on-stdin" in argv
            assert "--no-bar" in argv
    finally:
        window.close()


def test_an_explicit_output_directory_is_where_the_output_goes(
    qt_app: QApplication, tmp_path: pathlib.Path
) -> None:
    _source(tmp_path / "src")
    window = ComposeWindow()
    try:
        window.source_edit.setText(str(tmp_path / "src"))
        window.discover()
        window.output_edit.setText(str(tmp_path / "chosen"))

        window.video_radio.setChecked(True)
        argv = window.build_command()
        assert argv is not None
        target = pathlib.Path(argv[argv.index("--out") + 1])
        assert target.parent == tmp_path / "chosen"
        assert target.suffix == ".mp4" and target.stem.startswith("S_")

        window.frames_radio.setChecked(True)
        argv = window.build_command()
        assert argv is not None
        target = pathlib.Path(argv[argv.index("--out") + 1])
        assert target.parent == tmp_path / "chosen" and target.suffix == ""
    finally:
        window.close()


def test_the_option_lists_come_from_the_library(
    qt_app: QApplication, tmp_path: pathlib.Path
) -> None:
    """If these were typed into the .ui they would drift from the CLI the first time an
    option changed. They are read from `encode` and `stitch` instead."""
    from vr_compose import encode
    from vr_compose.stitch import BIT_DEPTHS, DEFAULT_SAMPLER, SAMPLERS

    window = ComposeWindow()
    try:

        def items(combo: QComboBox) -> list[str]:
            return [combo.itemText(i) for i in range(combo.count())]

        assert items(window.size_combo) == list(encode.SIZES)
        assert items(window.bitrate_combo) == list(encode.LADDER)
        assert items(window.sampler_combo) == list(SAMPLERS)
        assert items(window.bit_depth_combo) == [str(d) for d in BIT_DEPTHS]
        assert window.sampler_combo.currentText() == DEFAULT_SAMPLER
    finally:
        window.close()


def test_the_window_reports_progress_and_finishes(
    qt_app: QApplication, tmp_path: pathlib.Path
) -> None:
    """A real job, driven through the window, in a real subprocess.

    Small on purpose: this asserts the plumbing -- process starts, NDJSON parsed, bar
    advances, status ends on `pass` -- not the stitch, which the rest of the suite owns.
    """
    from PySide6.QtCore import QTimer

    _source(tmp_path / "src")
    window = ComposeWindow()
    try:
        window.source_edit.setText(str(tmp_path / "src"))
        window.discover()
        window.frames_edit.setText("1-3")
        window.frames_radio.setChecked(True)
        window.output_edit.setText(str(tmp_path / "out"))
        window.start()
        # bound to a local on purpose: asserting on `window.process` here would narrow the
        # attribute for the rest of the function, and mypy cannot see that the event loop
        # clears it, so the later `is None` check would look like dead code
        started = window.process
        assert started is not None, "a subprocess, not a thread"

        deadline = QTimer()
        deadline.timeout.connect(lambda: qt_app.quit() if window.process is None else None)
        deadline.start(100)
        QTimer.singleShot(300_000, qt_app.quit)
        qt_app.exec()

        assert window.process is None, "the job did not finish in time"
        status = str(window.status_banner.property("status"))
        assert status == "pass", window.log_edit.toPlainText()
        assert window.progress_bar.value() == window.progress_bar.maximum()
        assert str(window.progress_bar.property("state")) == "done"
        assert "s/帧" in window.progress_detail.text()

        made = sorted(p.name for p in (tmp_path / "out").rglob("*.png"))
        assert [name for name in made if name.startswith("S_S_")] == [
            "S_S_0001.png",
            "S_S_0002.png",
            "S_S_0003.png",
        ], made
    finally:
        window.close()


def test_worker_command_runs_this_interpreter() -> None:
    """From source the child is `python -m vr_compose`; frozen it is the exe itself."""
    command = worker_command()
    assert command[0] == sys.executable
    assert command[1:] == ["-m", "vr_compose"]
    assert len(command) == 3


# --- the packaging entry point ---------------------------------------------------------


def test_the_root_entry_point_is_both_faces(monkeypatch: pytest.MonkeyPatch) -> None:
    """A frozen build is one exe, and the window renders by launching *itself*.

    So `main_ui.py` has to be the CLI when given arguments and the window when not --
    otherwise pressing Start in a packaged build opens a second window instead of
    rendering. See `worker_command`, which returns `[sys.executable]` when frozen.
    """
    import main_ui

    assert main_ui.wants_cli([]) is False, "a double-click opens the window"
    assert main_ui.wants_cli(["--version"]) is True
    assert main_ui.wants_cli(["--source", "x", "sequence"]) is True

    called: list[tuple[str, list[str]]] = []

    def record(face: str) -> Callable[[list[str]], int]:
        def run(argv: list[str]) -> int:
            called.append((face, argv))
            return 0

        return run

    monkeypatch.setattr("vr_compose.cli.main", record("cli"))
    monkeypatch.setattr("vr_compose.gui.main", record("gui"))

    assert main_ui.main(["--source", "x", "discover"]) == 0
    assert called == [("cli", ["--source", "x", "discover"])]

    called.clear()
    assert main_ui.main([]) == 0
    assert [face for face, _ in called] == ["gui"], called


def test_the_entry_point_calls_freeze_support_first() -> None:
    """AGENTS.md constraint 4: without it every frozen pool worker starts a fresh GUI,
    recursively. Asserted on the source text because the failure only shows up in a
    packaged build, which the test suite cannot produce."""
    import inspect

    import main_ui

    source = inspect.getsource(main_ui)
    guard = source.index('if __name__ == "__main__":')
    body = [
        line.strip()
        for line in source[guard:].splitlines()[1:]
        if line.strip() and not line.strip().startswith("#")
    ]
    assert body[0] == "multiprocessing.freeze_support()", body


def test_the_entry_point_makes_the_src_layout_importable() -> None:
    """`python main_ui.py` in a checkout must work without installing the project."""
    import main_ui

    assert main_ui.SRC.name == "src"
    assert (main_ui.SRC / "vr_compose" / "__init__.py").is_file()
