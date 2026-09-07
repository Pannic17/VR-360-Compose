"""Support `python -m vr_compose`, and the frozen executable's entry point.

`freeze_support()` must run before anything else. A PyInstaller-frozen executable spawns
process-pool workers by re-executing itself, so without this call every worker would start
a fresh copy of the whole application -- recursively (AGENTS.md §2, constraint 4).
"""

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()

    from vr_compose.cli import main

    sys.exit(main())
