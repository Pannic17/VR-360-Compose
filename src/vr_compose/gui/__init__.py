"""PySide6 front end (ROADMAP P6). A shell over the CLI, never a second implementation.

Import is deliberately lazy: `vr_compose.gui.main` pulls Qt in only when it is called,
so importing this package costs nothing and a machine without PySide6 gets a sentence it
can act on rather than a traceback from the middle of a widget module.
"""

from __future__ import annotations

import sys

__all__ = ["main"]

_MISSING = """PySide6 is not installed, so the window cannot open.

    pip install "vr-compose[gui]"

The command line does not need it: `vr-compose sequence ...` does the same work, and the
GUI is only a shell over exactly that.
"""


def main(argv: list[str] | None = None) -> int:
    try:
        from vr_compose.gui.window import main as run
    except ImportError as exc:  # pragma: no cover -- needs a machine without PySide6
        if "PySide6" not in str(exc):
            raise
        print(_MISSING, file=sys.stderr)
        return 2
    return run(argv)
