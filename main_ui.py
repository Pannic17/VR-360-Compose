"""Root entry point for the packaged application. Point PyInstaller at this file.

It lives at the root, next to the future `.spec`, for the same reason VR-Installer's
`main_ui.py` does: a spec names a script, and a script at the root is the one place both
a checkout and a frozen build can agree on.

Four things it has to get right, none of which the `vr-compose-gui` console script can
do for a frozen build:

**One executable, two faces.** A frozen build is a single exe, and the window runs a
render by launching *itself* as a subprocess -- `vr_compose.gui.window.worker_command`
returns `[sys.executable]` when frozen. So the exe must behave as the CLI when it is
given CLI arguments and open the window when it is not. Without that, pressing Start in
a packaged build would open a second window instead of rendering. The rule is as plain
as it can be: **no arguments means the window, any arguments mean the command line.**

**`freeze_support()` first.** AGENTS.md section 2, constraint 4: after freezing, a
process pool derives workers by re-executing `sys.executable`, and without this call each
worker starts a whole fresh GUI -- recursively. The pipeline uses threads today, so
nothing triggers it *yet*; it is the first statement anyway, because the failure mode is
a fork bomb wearing the application's icon and the fix has to already be there when
someone adds a pool.

**The src layout.** `vr_compose` lives under `src/`, so a plain `python main_ui.py` in a
checkout cannot import it unless the project is installed. Adding `src/` when it is
there costs nothing and makes this file work either way, which is what you want from the
script a build is aimed at.

**No black window on a double-click**, which the two build shapes reach differently and
for measured reasons (2026-09-09, after the user reported one):

* the **single-file** build is *windowed*, so it has no console at all. Hiding one was
  tried first and cannot work there: the console belongs to the *bootloader* and is on
  screen before this code runs, i.e. for the whole 494 MiB unpack. A 70 MiB stand-in
  still showed it for about four seconds, and hiding it then left a taskbar button.
* the **folder** build keeps its console and `_hide_own_console` hides it. That build
  starts in 0.35 s, so the console goes before anyone sees it, and keeping it preserves
  what a windowed build gives up: **PowerShell waits for a console process.** Typed bare
  into PowerShell, a windowed `VR-Compose.exe --version` prints nothing and leaves no exit
  code, because the shell does not wait for a GUI-subsystem process (through a pipe, or
  `cmd`, or `Start-Process -Wait`, it is correct). The shape that gets scripted keeps its
  console; the shape that gets double-clicked has none.

Both faces work either way. A frozen windowed process does *not* simply lose its streams:
Windows attaches a new process to its parent's console whatever the subsystem says -- the
subsystem only decides whether a *new* console is allocated when there is no parent one --
and the window's render subprocess is handed pipes, which measured intact. What is left is
a start with no streams at all, which `_attach_parent_console` covers.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import pathlib
import sys
from collections.abc import Callable, Sequence
from typing import TextIO, cast

SRC = pathlib.Path(__file__).resolve().parent / "src"


def _make_package_importable() -> None:
    """Put `src/` on the path when running from a checkout that is not installed.

    A frozen build has no `src/` beside the executable and does not need one -- the
    package is inside the bundle -- so this quietly does nothing there.
    """
    if SRC.is_dir() and str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))


def needs_streams(stdout: object, stderr: object) -> bool:
    """Does the command-line face have nowhere to write?

    True only when a stream is missing. Whoever started us usually supplies them: a
    terminal passes its console handles down, and the window passes *pipes* to the render
    it launches -- and that pipe is the progress channel (NDJSON, read by `QProcess`), so
    touching the streams in that case would send progress somewhere nobody is reading.
    """
    return stdout is None or stderr is None


def _attach_parent_console() -> None:
    """Make sure the command-line face has somewhere to print. Usually a no-op.

    The single-file build is *windowed*, which is what keeps a double-click from putting a
    console on the screen at all. That costs nothing in a terminal: Windows attaches a new
    process to its parent's console whatever the subsystem says, so the streams are already
    there and :func:`needs_streams` says no (measured -- `AttachConsole` then fails
    precisely because this process is attached already).

    What remains is the case with no streams *and* no console: started detached, or from
    Explorer with arguments. Then this borrows the parent's console if there is one, and
    failing that binds the streams to nowhere -- because the alternative is that the first
    `print` in the CLI face raises `AttributeError` on `None` and the run dies for the sake
    of a message no one was going to see.
    """
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return
    if not needs_streams(sys.stdout, sys.stderr):
        return
    try:
        import ctypes

        if not ctypes.windll.kernel32.AttachConsole(-1):  # ATTACH_PARENT_PROCESS
            _silence_missing_streams()
            return

        # `CONOUT$` / `CONIN$` name the console we have just attached to, whatever it is.
        # These stay open for the life of the process on purpose -- they *are* its
        # streams -- so a context manager would close what the rest of the run needs.
        # Line buffered, so a long render's log appears as it happens.
        def console_stream(name: str, mode: str) -> TextIO:
            opened = open(name, mode, encoding="utf-8", errors="replace", buffering=1)  # noqa: SIM115
            return cast("TextIO", opened)

        sys.stdout = console_stream("CONOUT$", "w")
        sys.stderr = console_stream("CONOUT$", "w")
        with contextlib.suppress(OSError):
            sys.stdin = console_stream("CONIN$", "r")
    except Exception:
        # Without a console the CLI face prints nothing, which is a poor answer but a
        # working one. Failing to start would be worse.
        _silence_missing_streams()


def _silence_missing_streams() -> None:
    """Point any missing stream at nowhere, so printing cannot raise."""
    for name in ("stdout", "stderr"):
        if getattr(sys, name, None) is None:
            with contextlib.suppress(OSError):
                setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))  # noqa: SIM115


def console_is_only_ours(pids: Sequence[int], image_of: Callable[[int], str | None]) -> bool:
    """Is every process sharing this console running *our* executable?

    The folder build keeps its console (see the module docstring) and hides it for a
    double-click, and this is the test for "the console was made for us". Counting the
    processes is *not* the same test: a onefile build is a bootloader plus its child, so a
    `== 1` check answered no for it -- which is the bug this replaced, before the onefile
    shape stopped having a console at all.

    * folder build, double-clicked -- `[us]`, so ours;
    * either shape from a terminal -- the shell is in the list and is not our executable,
      so the console is somebody else's and must be left alone. Hiding a shell's window
      would take the terminal away from whoever is using it.

    An image we cannot read (a pid that has just exited, or a process we may not open)
    counts as somebody else's: an unnecessary black rectangle is a blemish, and hiding a
    terminal is a fault.
    """
    if not pids:
        return False
    mine = image_of(os.getpid())
    return mine is not None and all(image_of(pid) == mine for pid in pids)


def _console_pids(kernel32: object) -> list[int]:
    """The pids attached to this console. Empty means "could not tell"."""
    import ctypes

    for _attempt in range(2):
        # The first call asks how many there are: given too small a buffer the function
        # fills nothing and returns the size it needs.
        needed = kernel32.GetConsoleProcessList((ctypes.c_uint32 * 1)(), 1)  # type: ignore[attr-defined]
        if needed <= 0:
            return []
        buffer = (ctypes.c_uint32 * needed)()
        count = kernel32.GetConsoleProcessList(buffer, needed)  # type: ignore[attr-defined]
        if 0 < count <= needed:
            return list(buffer[:count])
        # A process attached between the two calls; ask again with the new size.
    return []


def _process_image(pid: int) -> str | None:
    """The full path of a running process's executable, case-folded, or None."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    # QUERY_LIMITED_INFORMATION rather than QUERY_INFORMATION: it is the right that works
    # across integrity levels, so an elevated shell in the list is still identifiable.
    handle = kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return None
    try:
        length = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(length.value)
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(length)):
            return None
        return buffer.value.casefold()
    finally:
        kernel32.CloseHandle(handle)


def _hide_own_console() -> None:
    """Hide the console window, if this application is the only thing using it.

    Only the *folder* build has a console to hide -- the single-file build is windowed, so
    here `GetConsoleWindow` returns nothing and this returns immediately. The folder build
    starts in about 0.35 s, so its console is gone before it can be noticed; the same trick
    was useless for the single-file build, whose bootloader shows the console for the
    entire unpack, which is why that shape has none.

    Hiding rather than `FreeConsole` is deliberate: the window renders by launching this
    same executable as a child, and a child console application inherits this console. A
    hidden console is inherited hidden, so no black window appears when a render starts;
    a freed one would make Windows give the child a brand new, visible one.

    Set `VRC_CONSOLE_LOG` to a file path to find out what this decided and why. It is the
    only way to see: the decision depends on being a *packaged* process, so it cannot be
    reproduced from a checkout, and this swallows its exceptions on purpose -- which means
    that without the log, a wrong answer is perfectly silent.
    """
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return
    note = _console_log
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetConsoleWindow.restype = wintypes.HWND
        window = kernel32.GetConsoleWindow()
        note(f"pid={os.getpid()} image={_process_image(os.getpid())} window={window}")
        if not window:
            note("no console window: nothing to hide (the windowed build)")
            return
        pids = _console_pids(kernel32)
        note(f"console pids={pids} images={[_process_image(pid) for pid in pids]}")
        if console_is_only_ours(pids, _process_image):
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
            user32.ShowWindow.restype = wintypes.BOOL
            was_visible = user32.ShowWindow(window, 0)  # SW_HIDE
            note(f"hid the console (it was visible: {bool(was_visible)})")
        else:
            note("the console is somebody else's; left alone")
    except Exception as exc:
        # A cosmetic tweak must never be the thing that stops the application starting.
        note(f"failed: {type(exc).__name__}: {exc}")


def _console_log(message: str) -> None:
    """Append a diagnosis line to `%VRC_CONSOLE_LOG%`, if that is set. Never raises."""
    path = os.environ.get("VRC_CONSOLE_LOG")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(message + "\n")
    except Exception:
        # Diagnostics must not break the thing they are diagnosing.
        pass


def wants_cli(argv: list[str]) -> bool:
    """Should this invocation be the command line rather than the window?

    Any argument at all means yes. That covers what the window itself passes when it
    launches a render (`--source ... sequence ...`), and it keeps `--help` and `--version`
    working on the packaged exe, which matters when someone is trying to work out what
    they have got. Double-clicking passes nothing, so a double-click opens the window.
    """
    return bool(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    _make_package_importable()
    if wants_cli(arguments):
        _attach_parent_console()
        from vr_compose.cli import main as run_cli

        return run_cli(arguments)
    # The window has nowhere to print either, and something in the stack eventually will
    # -- a Qt warning, a library's `print`. Missing streams make that an AttributeError
    # instead of a discarded line, so give them somewhere to go and get on with it.
    _silence_missing_streams()
    _hide_own_console()
    from vr_compose.gui import main as run_gui

    return run_gui([sys.argv[0]])


if __name__ == "__main__":
    # First statement, deliberately: see the module docstring and AGENTS.md constraint 4.
    multiprocessing.freeze_support()
    sys.exit(main())
