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

A fourth thing is cosmetic but visible: the build must be a *console* application,
because a windowed PyInstaller build leaves `sys.stdout` as None and both faces write to
it. `_hide_own_console` therefore hides the console when -- and only when -- this process
is the sole owner of it, which is what a double-click looks like.
"""

from __future__ import annotations

import multiprocessing
import pathlib
import sys

SRC = pathlib.Path(__file__).resolve().parent / "src"


def _make_package_importable() -> None:
    """Put `src/` on the path when running from a checkout that is not installed.

    A frozen build has no `src/` beside the executable and does not need one -- the
    package is inside the bundle -- so this quietly does nothing there.
    """
    if SRC.is_dir() and str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))


def _hide_own_console() -> None:
    """Hide the console window, but only if this process is the one it was made for.

    The build has to be a console application: a windowed PyInstaller build leaves
    `sys.stdout` as None, which would break both faces at once -- the CLI would print
    nothing, and the window's NDJSON pipe writes to stdout. So the console exists, and a
    double-click would otherwise show an empty black rectangle behind the window.

    `GetConsoleProcessList` says how many processes share this console. One means it was
    created for us and nobody else is using it, so hiding it is safe. More than one means
    we were launched from somebody's terminal, and hiding *that* would be rude -- so this
    does nothing, which is also what you want when a render is being driven from a shell.
    """
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        window = kernel32.GetConsoleWindow()
        if not window:
            return
        buffer = (ctypes.c_uint32 * 2)()
        if kernel32.GetConsoleProcessList(buffer, 2) == 1:
            ctypes.windll.user32.ShowWindow(window, 0)  # SW_HIDE
    except Exception:
        # A cosmetic tweak must never be the thing that stops the application starting.
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
        from vr_compose.cli import main as run_cli

        return run_cli(arguments)
    _hide_own_console()
    from vr_compose.gui import main as run_gui

    return run_gui([sys.argv[0]])


if __name__ == "__main__":
    # First statement, deliberately: see the module docstring and AGENTS.md constraint 4.
    multiprocessing.freeze_support()
    sys.exit(main())
