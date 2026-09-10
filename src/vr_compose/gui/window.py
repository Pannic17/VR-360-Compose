"""The P6 window: a thin shell that drives the CLI in a separate process.

Structure and look follow VR-Installer's `main_ui.py` -- a `.ui` loaded at runtime with
`QUiLoader`, widgets bound by object name with `findChild`, a `.qss` applied to the
window, and `variant` / `status` dynamic properties carrying the styling states. One
thing is deliberately *not* copied: that tool runs its work in a `QThread`, and AGENTS.md
section 7 requires this one to run the job in its own **process**.

The reasons are specific rather than stylistic. The pipeline already owns a decode thread
pool, a warp thread pool and a chain of ffmpeg children; putting all of that in the
interpreter that is also running Qt's event loop invites exactly the interference the
constraint is written to avoid. A separate process also means a crash in the job closes a
subprocess instead of the window, and cancellation gets a channel that cannot deadlock on
the GIL.

So the window builds an argv, runs `vr_compose.cli` with `--progress-json
--cancel-on-stdin`, and reads NDJSON off the pipe. `QProcess` does that on the event
loop, so no thread is needed here at all, and the guarantees on cancellation are the
pipeline's own: finished segments or frames are kept, no partial files, resumable.

**There is no second implementation of anything.** Discovery is `vr_compose.source`, the
rig comes from `vr_compose.rig`, the option lists come from `encode` and `stitch`, and
the job is the CLI. This module contains layout, argv assembly and event handling.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

from PySide6.QtCore import QObject, QProcess, Qt, Slot
from PySide6.QtGui import QIcon, QTextCursor
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QWidget,
)

from vr_compose import __version__, encode, pipeline
from vr_compose import source as source_mod
from vr_compose.device import GUI_DEVICE
from vr_compose.harmonise import DEFAULT_HARMONISE
from vr_compose.rig import UnknownRigError, rig_for
from vr_compose.stitch import BIT_DEPTHS, DEFAULT_SAMPLER, SAMPLERS

HERE = pathlib.Path(__file__).resolve().parent
UI_FILE = HERE / "main_window.ui"
QSS_FILE = HERE / "style.qss"
ICON_FILE = HERE / "icon.png"
"""The application icon (user, 2026-09-09). The executable carries the same picture as
`icon.ico` through the PyInstaller spec; this copy is for the window and the taskbar."""

CHOOSE_STEM = "— 请选择 —"
"""Placeholder shown when a directory holds several renders.

A "stem" is the prefix UE gave the files -- `L_Cathedral` in `L_Cathedral.1656.png` --
so it is the level or sequence name. It has nothing to do with left and right eyes
(`L_` is UE's Level prefix), and one is present in every source set, single or not: it
is the name every output is built from. The control only asks a question when one set of
`Camera*` directories holds more than one render, which is why the label reads 渲染名.

ROADMAP P6 item 6: with more than one stem the user picks, and the tool does not quietly
take the first. A combo box always has something selected, so the something is this, and
Start stays disabled until it is replaced.
"""


def refresh_style(widget: QWidget) -> None:
    """Re-evaluate the stylesheet after a dynamic property changed."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)


def worker_command() -> list[str]:
    """How to launch the CLI as a child process.

    From source that is `python -m vr_compose`. Frozen, the executable *is* the
    application, so it is invoked directly and P8 has to make the packaged entry point
    dispatch to `vr_compose.cli` when it sees CLI arguments -- worth knowing now, while
    the coupling is one function long.
    """
    if getattr(sys, "frozen", False):
        return [sys.executable]
    return [sys.executable, "-m", "vr_compose"]


class ComposeWindow(QMainWindow):
    """Choose a source, choose parameters, run, watch, cancel."""

    def __init__(self) -> None:
        super().__init__()
        self.process: QProcess | None = None
        self.candidates: list[source_mod.SourceSet] = []
        self.total_frames = 0
        self._cancelling = False

        loaded = self._load_ui()
        self.base_title = f"{loaded.windowTitle()}  {__version__}"
        self.setWindowTitle(self.base_title)
        self.setWindowIcon(QIcon(str(ICON_FILE)))
        self.resize(loaded.size())
        central = loaded.takeCentralWidget()
        if central is None:
            raise RuntimeError(f"{UI_FILE} has no central widget")
        self.setCentralWidget(central)
        self._bind_widgets()
        self._load_style()
        self._apply_static_properties()
        self._populate_choices()
        self.discover()

    # --- construction -----------------------------------------------------------------

    def _load_ui(self) -> QMainWindow:
        loader = QUiLoader()
        ui = loader.load(str(UI_FILE), self)
        if ui is None:
            raise RuntimeError(f"could not load the UI file: {UI_FILE}")
        assert isinstance(ui, QMainWindow)
        return ui

    def _load_style(self) -> None:
        if QSS_FILE.is_file():
            self.setStyleSheet(QSS_FILE.read_text(encoding="utf-8"))

    def _child(self, kind: type[QObject], name: str) -> Any:
        widget = self.findChild(kind, name)
        if widget is None:
            raise RuntimeError(f"the UI file has no {kind.__name__} called {name!r}")
        return widget

    def _bind_widgets(self) -> None:
        self.status_banner: QLabel = self._child(QLabel, "statusBannerLabel")
        self.source_edit: QLineEdit = self._child(QLineEdit, "sourceLineEdit")
        self.browse_source: QPushButton = self._child(QPushButton, "browseSourceButton")
        self.stem_combo: QComboBox = self._child(QComboBox, "stemComboBox")
        self.frames_edit: QLineEdit = self._child(QLineEdit, "framesLineEdit")
        self.output_edit: QLineEdit = self._child(QLineEdit, "outputLineEdit")
        self.browse_output: QPushButton = self._child(QPushButton, "browseOutputButton")
        self.video_radio: QRadioButton = self._child(QRadioButton, "videoRadioButton")
        self.frames_radio: QRadioButton = self._child(QRadioButton, "framesRadioButton")
        self.size_combo: QComboBox = self._child(QComboBox, "sizeComboBox")
        self.codec_combo: QComboBox = self._child(QComboBox, "codecComboBox")
        self.bitrate_combo: QComboBox = self._child(QComboBox, "bitrateComboBox")
        self.fps_combo: QComboBox = self._child(QComboBox, "fpsComboBox")
        self.sampler_combo: QComboBox = self._child(QComboBox, "samplerComboBox")
        self.harmonise_check: QCheckBox = self._child(QCheckBox, "harmoniseCheckBox")
        self.bit_depth_combo: QComboBox = self._child(QComboBox, "bitDepthComboBox")
        self.progress_bar: QProgressBar = self._child(QProgressBar, "progressBar")
        self.progress_detail: QLabel = self._child(QLabel, "progressDetailLabel")
        self.log_edit: QPlainTextEdit = self._child(QPlainTextEdit, "logTextEdit")
        self.discover_button: QPushButton = self._child(QPushButton, "discoverButton")
        self.cancel_button: QPushButton = self._child(QPushButton, "cancelButton")
        self.start_button: QPushButton = self._child(QPushButton, "startButton")

        self.browse_source.clicked.connect(self.choose_source)
        self.browse_output.clicked.connect(self.choose_output)
        self.discover_button.clicked.connect(self.discover)
        self.start_button.clicked.connect(self.start)
        self.cancel_button.clicked.connect(self.cancel)
        self.stem_combo.currentTextChanged.connect(self._stem_changed)
        self.video_radio.toggled.connect(self._mode_changed)

    def _apply_static_properties(self) -> None:
        for button, variant in (
            (self.browse_source, "outline"),
            (self.browse_output, "outline"),
            (self.discover_button, "outline"),
            (self.cancel_button, "ghost"),
            (self.start_button, "primary"),
        ):
            button.setProperty("variant", variant)
            refresh_style(button)

    def _populate_choices(self) -> None:
        """Every option list comes from the library, so the GUI cannot drift from the CLI."""
        self.size_combo.addItems(list(encode.SIZES))
        self.codec_combo.addItems(["h264", "h265"])
        self.bitrate_combo.addItems(list(encode.LADDER))
        self.fps_combo.addItems(["30", "60"])
        self.sampler_combo.addItems(list(SAMPLERS))
        self.sampler_combo.setCurrentText(DEFAULT_SAMPLER)
        self.harmonise_check.setChecked(DEFAULT_HARMONISE)
        self.bit_depth_combo.addItems([str(depth) for depth in BIT_DEPTHS])
        self._mode_changed()

    # --- discovery --------------------------------------------------------------------

    @Slot()
    def discover(self) -> None:
        """Find the source sets, and report a rejected directory in the words it used.

        ROADMAP P6 item 5: a directory that does not qualify must say *why* -- "tiles are
        not square", "camera numbering has gaps" -- rather than "invalid directory". Those
        sentences already exist on `SourceSet.problems`; this just shows them.
        """
        text = self.source_edit.text().strip()
        explicit = pathlib.Path(text) if text else None
        found, searched = source_mod.discover(explicit)
        self.candidates = found

        if not found:
            self.log(f"没有找到可用的来源集，搜索了 {len(searched)} 个位置：")
            for path in searched[:8]:
                self.log(f"  {path}")
            for rejected in source_mod.scan(explicit) if explicit else []:
                self.log(f"被拒绝的 stem {rejected.stem!r}：")
                for problem in rejected.problems:
                    self.log(f"  - {problem}")
            self.log(
                "一个来源目录至少要有两个 `CameraN` 子目录，里面的 PNG 名为 "
                "`<stem>.<帧号>.png`（直接放在里面或再下一层都行）。"
            )
            self._set_stems([])
            self.set_status("fail", "无来源")
            return

        if not text:
            self.source_edit.setText(str(found[0].root))
        self._set_stems(found)
        self.log(f"在 {found[0].root} 找到 {len(found)} 个来源集。")

    def _set_stems(self, found: list[source_mod.SourceSet]) -> None:
        self.stem_combo.blockSignals(True)
        self.stem_combo.clear()
        if len(found) > 1:
            # more than one scene: make the user pick (P6 item 6)
            self.stem_combo.addItem(CHOOSE_STEM)
            self.stem_combo.addItems([s.stem for s in found])
            names = "、".join(s.stem for s in found)
            self.log(
                f"这个目录里有 {len(found)} 套渲染（文件名前缀：{names}），请选择要处理的那个。"
            )
        else:
            self.stem_combo.addItems([s.stem for s in found])
        self.stem_combo.setEnabled(bool(found))
        self.stem_combo.blockSignals(False)
        self._stem_changed(self.stem_combo.currentText())

    def selected(self) -> source_mod.SourceSet | None:
        stem = self.stem_combo.currentText()
        if not stem or stem == CHOOSE_STEM:
            return None
        return next((s for s in self.candidates if s.stem == stem), None)

    @Slot(str)
    def _stem_changed(self, _stem: str = "") -> None:
        chosen = self.selected()
        self.start_button.setEnabled(chosen is not None and self.process is None)
        if chosen is None:
            self.progress_detail.setText("—")
            return
        try:
            rig = rig_for(chosen.camera_count)
            tile = chosen.tile_size
            native = rig.native_width(tile) if tile else 0
            frames = chosen.frames
            self.progress_detail.setText(
                f"{chosen.camera_count} 路 · tile {tile}px · 母版 {native}x{native // 2} · "
                f"共有帧 {frames[0]}..{frames[-1]}（{len(frames)} 帧）"
                if frames
                else "没有所有相机都具备的帧"
            )
            if not self.frames_edit.text().strip() and frames:
                # The last contiguous run, not everything: the reference set carries a
                # stray 0000 frame, and offering "0-2433" would prefill the exact trap
                # the docs warn about (pipeline.contiguous_tail explains it).
                first, last = pipeline.contiguous_tail(frames)
                self.frames_edit.setText(f"{first}-{last}")
                if first != frames[0]:
                    skipped = len(frames) - (last - first + 1)
                    self.log(
                        f"帧范围填的是最后一段连续帧 {first}-{last}；"
                        f"前面有 {skipped} 帧与它不连续（参考数据里那张孤立的 0000），"
                        "把它算进去会让视频开头跳帧。要全部渲染就自己改成 all。"
                    )
            self.set_status("pending", "就绪")
        except UnknownRigError as exc:
            self.log(f"装配不认识：{exc}")
            self.start_button.setEnabled(False)
            self.set_status("fail", "装配未登记")

    @Slot()
    def _mode_changed(self, _checked: bool = False) -> None:
        """Grey out what the chosen mode cannot use, because the CLI refuses it outright.

        Frame mode has no codec, bitrate, frame rate or delivery size -- it writes at the
        input's own density, losslessly (P5) -- and video mode has no master bit depth.
        Showing that as disabled controls is friendlier than letting the CLI reject the
        combination after the fact.
        """
        video = self.video_radio.isChecked()
        for widget in (self.size_combo, self.codec_combo, self.bitrate_combo, self.fps_combo):
            widget.setEnabled(video)
        self.bit_depth_combo.setEnabled(not video)

    # --- browsing ---------------------------------------------------------------------

    @Slot()
    def choose_source(self) -> None:
        start = self.source_edit.text().strip() or str(pipeline.default_output_dir())
        picked = QFileDialog.getExistingDirectory(self, "选择来源目录", start)
        if picked:
            self.source_edit.setText(picked)
            self.discover()

    @Slot()
    def choose_output(self) -> None:
        """Both modes want a directory: frame mode fills one, video mode is named inside one."""
        start = self.output_edit.text().strip() or str(pipeline.default_output_dir())
        picked = QFileDialog.getExistingDirectory(self, "选择输出位置", start)
        if picked:
            self.output_edit.setText(picked)

    # --- running ----------------------------------------------------------------------

    def build_command(self) -> list[str] | None:
        """Assemble the CLI argv, or None when nothing is selected yet.

        Deliberately silent: it shows no dialog and touches no state, so "what would this
        window run?" is a question a test can ask. :meth:`start` is where the user gets
        told. (Written the other way round first, and the modal made it untestable.)
        """
        chosen = self.selected()
        if chosen is None:
            return None
        frames = self.frames_edit.text().strip() or "all"
        video = self.video_radio.isChecked()

        argv = [
            *worker_command(),
            "--source",
            str(chosen.root),
            "sequence",
            "--stem",
            chosen.stem,
            "--frames",
            frames,
            "--sampler",
            self.sampler_combo.currentText(),
            # The GUI has no device control (user, 2026-09-09): it asks for the fastest
            # device the machine has, and the CLI falls back to the CPU quietly.
            "--device",
            GUI_DEVICE,
            "--no-bar",
            "--progress-json",
            "--cancel-on-stdin",
        ]
        if not self.harmonise_check.isChecked():
            argv.append("--no-harmonise")  # the default is the library's; only say "no"
        if video:
            argv += [
                "--size",
                self.size_combo.currentText(),
                "--codec",
                self.codec_combo.currentText(),
                "--bitrate",
                self.bitrate_combo.currentText(),
                "--fps",
                self.fps_combo.currentText(),
            ]
        else:
            argv += ["--out-format", "png", "--bit-depth", self.bit_depth_combo.currentText()]

        target = self.output_edit.text().strip()
        if target:
            directory = pathlib.Path(target)
            stamp = pipeline.run_stamp()
            # A directory was chosen, but the video path wants a file inside it. Naming it
            # here rather than letting the CLI default keeps the output where the user
            # pointed, and keeps resume possible: the path is printed and reusable.
            # `unique_path` because this name is auto-generated even though `--out` is
            # explicit: the user picked a *directory*, not a name, so the P5 promise that
            # an auto-generated name is a fresh output still applies. Minute-resolution
            # stamps make two runs in one minute collide otherwise.
            argv += [
                "--out",
                str(
                    pipeline.unique_path(
                        directory / pipeline.default_master_dir_name(chosen, stamp)
                        if not video
                        else directory / pipeline.default_output_name(chosen, stamp)
                    )
                ),
            ]
        return argv

    @Slot()
    def start(self) -> None:
        argv = self.build_command()
        if argv is None:
            QMessageBox.warning(self, "还差一步", "请先选择一个来源集。")
            return
        self.log_edit.clear()
        self.log("$ " + " ".join(argv))
        self.total_frames = 0
        self._cancelling = False
        self.progress_bar.setProperty("state", "")
        # Busy (indeterminate) until the `start` event says how many frames there are.
        # Discovery, the warp plan and the disk pre-check happen first and take a few
        # seconds; a bar frozen at 0% for that long reads as "nothing is happening".
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("准备中…")
        refresh_style(self.progress_bar)
        self.progress_detail.setText("正在发现来源、构建 warp 表、预检磁盘…")
        self.set_status("running", "运行中")
        self.set_running(True)

        process = QProcess(self)
        process.setProgram(argv[0])
        process.setArguments(argv[1:])
        # stdout is the NDJSON stream and stderr is prose; keeping them apart is the whole
        # point of --progress-json, so they must not be merged into one channel.
        process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        process.readyReadStandardOutput.connect(self._read_events)
        process.readyReadStandardError.connect(self._read_prose)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(self._process_error)
        self.process = process
        process.start()

    @Slot()
    def cancel(self) -> None:
        """Ask the job to stop. A line on stdin is all it takes (`--cancel-on-stdin`)."""
        if self.process is None:
            return
        self._cancelling = True
        self.cancel_button.setEnabled(False)
        self.set_status("pending", "正在停止")
        self.log("请求取消：已完成的部分会保留，可以续跑。")
        self.process.write(b"cancel\n")

    @Slot()
    def _read_events(self) -> None:
        assert self.process is not None
        while self.process.canReadLine():
            line = bytes(self.process.readLine().data()).decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                self.log(line)  # not ours; show it rather than swallow it
                continue
            self._handle_event(event)

    def _handle_event(self, event: dict[str, Any]) -> None:
        kind = event.get("event")
        if kind == "start":
            self.total_frames = int(event.get("total", 0))
            self.progress_bar.setRange(0, max(self.total_frames, 1))
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("%p%   ·   %v / %m 帧")
            self.log(f"开始：{event.get('describe', '')} → {event.get('output', '')}")
        elif kind == "progress":
            done = int(event.get("done", 0))
            self.progress_bar.setValue(done)
            # Also in the title, which is what the taskbar shows: an 8K job runs for
            # hours and nobody watches the window for all of it.
            total = max(self.total_frames, 1)
            self.setWindowTitle(f"{done * 100 // total}%  ·  {self.base_title}")
            eta = float(event.get("eta_seconds", 0.0))
            self.progress_detail.setText(
                f"帧 {event.get('frame')}   {event.get('done')}/{event.get('total')}   "
                f"段 {event.get('segment')}/{event.get('segments')}   "
                f"{event.get('seconds_per_frame'):.2f} s/帧"
                f"（{event.get('speedup')}× vs 46.85）   剩余 {eta / 60:.1f} 分钟"
            )
        elif kind == "log":
            self.log(str(event.get("message", "")))
        elif kind == "cancelled":
            self.log(f"已取消：{event.get('message', '')}")
        elif kind == "error":
            self.log(f"失败：{event.get('message', '')}")
        elif kind == "done":
            self._report_done(event)

    def _report_done(self, event: dict[str, Any]) -> None:
        size = int(event.get("size_bytes", 0)) / 2**30
        self.log(
            f"完成：{event.get('output')}  ({size:.2f} GiB, {event.get('seconds_per_frame')} s/帧)"
        )
        for problem in event.get("problems", []):
            self.log(f"  合规问题：{problem}")
        for warning in event.get("warnings", []):
            self.log(f"  警告：{warning}")

    @Slot()
    def _read_prose(self) -> None:
        assert self.process is not None
        text = bytes(self.process.readAllStandardError().data()).decode("utf-8", "replace")
        for line in text.splitlines():
            if line.strip():
                self.log(line.rstrip())

    @Slot()
    def _process_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self.log(f"无法启动子进程：{' '.join(worker_command())}")
            self._leave_busy()
            self.setWindowTitle(self.base_title)
            self.set_status("fail", "启动失败")
            self.set_running(False)
            self.process = None

    def _leave_busy(self) -> None:
        """Take the bar out of indeterminate mode.

        Needed on every exit path: a job that fails during the pre-render phase never
        sends a `start` event, and a bar left sweeping forever after the run is over is
        the one thing worse than a bar that does not move.
        """
        if self.progress_bar.maximum() == 0:
            self.progress_bar.setRange(0, 1)
        self.progress_bar.setFormat("%p%")

    @Slot()
    def _finished(self, code: int, _status: QProcess.ExitStatus) -> None:
        self.process = None
        self.set_running(False)
        self._leave_busy()
        self.setWindowTitle(self.base_title)
        if code == 0:
            self.progress_bar.setValue(self.progress_bar.maximum())
            self.progress_bar.setProperty("state", "done")
            self.set_status("pass", "完成")
        elif code == 130 or self._cancelling:
            self.progress_bar.setProperty("state", "")
            self.set_status("pending", "已取消")
        else:
            self.progress_bar.setProperty("state", "fail")
            self.set_status("fail", f"失败（退出码 {code}）")
            QMessageBox.critical(self, "作业失败", f"子进程以退出码 {code} 结束。详情见日志。")
        refresh_style(self.progress_bar)

    # --- small helpers ----------------------------------------------------------------

    def log(self, text: str) -> None:
        self.log_edit.moveCursor(QTextCursor.MoveOperation.End)
        self.log_edit.insertPlainText(text + "\n")
        self.log_edit.moveCursor(QTextCursor.MoveOperation.End)

    def set_status(self, status: str, text: str) -> None:
        self.status_banner.setText(text)
        self.status_banner.setProperty("status", status)
        refresh_style(self.status_banner)

    def set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running and self.selected() is not None)
        self.cancel_button.setEnabled(running)
        for widget in (
            self.source_edit,
            self.browse_source,
            self.browse_output,
            self.discover_button,
            self.stem_combo,
            self.frames_edit,
            self.output_edit,
            self.video_radio,
            self.frames_radio,
            self.sampler_combo,
        ):
            widget.setEnabled(not running)
        if not running:
            self._mode_changed()
        else:
            for widget in (
                self.size_combo,
                self.codec_combo,
                self.bitrate_combo,
                self.fps_combo,
                self.bit_depth_combo,
            ):
                widget.setEnabled(False)

    def closeEvent(self, event: Any) -> None:
        """Closing the window stops the job -- politely first, then not.

        Closing stdin is itself a cancel request (`--cancel-on-stdin` treats EOF as one),
        so a job that respects it shuts down cleanly and stays resumable. `kill` is the
        fallback for one that does not.
        """
        process = self.process
        if process is not None and process.state() != QProcess.ProcessState.NotRunning:
            process.write(b"cancel\n")
            process.closeWriteChannel()
            if not process.waitForFinished(5000):
                process.kill()
                process.waitForFinished(2000)
        event.accept()


def main(argv: list[str] | None = None) -> int:
    """Entry point for `vr-compose-gui` and `python -m vr_compose.gui`."""
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("VR-Compose")
    app.setApplicationVersion(__version__)
    app.setWindowIcon(QIcon(str(ICON_FILE)))
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontShowIconsInMenus, True)
    window = ComposeWindow()
    window.show()
    return int(app.exec())
