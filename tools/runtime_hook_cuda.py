"""Make the bundle look like a CUDA toolkit, before anything imports cupy.

Only in the `VRC_GPU=1` build -- VR-Compose.spec adds this as a PyInstaller runtime hook
there and nowhere else. That build carries `bin/` (three libraries) and `include/` (the
toolkit's headers), and three separate lookups have to find them:

* cuda-pathfinder locates nvrtc with a native `LoadLibraryExW`, which searches the
  directories given to `os.add_dll_directory` -- PyInstaller adds the bundle root, not a
  subdirectory of it, so `bin/` has to be added here;
* `cupy._environment` takes the grandparent of wherever nvrtc was found and adds
  `<that>/bin` itself, which is why the libraries live in a `bin/` at all;
* NVRTC needs the headers when it compiles a kernel, and pathfinder looks for them under
  `$CUDA_PATH/include`. Without that variable it tries to find a CUDA installation by
  running `sys.executable -m cuda.pathfinder...`, and in a frozen application
  `sys.executable` is this program: it reads the module path as a subcommand, argparse
  rejects it, and the render dies with `ChildProcessError` (measured 2026-09-10).
* NVRTC then has to find **its own** `nvrtc-builtins64_<version>.dll`, and it does that
  through the classic library search, which reads `PATH` and does *not* read the list
  `os.add_dll_directory` maintains. So `bin/` goes on `PATH` as well -- belt and braces
  that turn out to be two different straps. Measured 2026-09-11, and the version matters:
  the toolkit's 12.4 NVRTC found its builtins beside itself with no help, the 12.9 wheel
  does not, and the whole render fails with `nvrtc: error: failed to open
  nvrtc-builtins64_129.dll` -- which the callers turn into a CPU fallback, so the only
  symptom is a slow render.

**`CUDA_PATH` is set, not defaulted.** A machine with its own toolkit would otherwise
send NVRTC to *those* headers, which may be a different CUDA version than the libraries
in `bin/`; the bundle is self-contained on purpose and should behave the same everywhere.

Nothing is set when `bin/` is absent, which is every build but the GPU one. The failure
mode being guarded against is silent: cupy fails to import, the device gate reads that as
"no GPU", and the job runs on the CPU without saying anything (`device`, rule 4).
"""

import os
import sys

_bundle = getattr(sys, "_MEIPASS", None)
if _bundle and sys.platform == "win32":
    # `_MEIPASS` is the directory holding the bundled files, which for a folder build is
    # `_internal`. Both spellings are tried rather than asserting which one this
    # PyInstaller uses -- the cost of being wrong is a CPU fallback nobody notices.
    for _root in (_bundle, os.path.join(_bundle, "_internal")):
        _libraries = os.path.join(_root, "bin")
        if os.path.isdir(_libraries):
            os.add_dll_directory(_libraries)
            os.environ["CUDA_PATH"] = _root
            os.environ["PATH"] = _libraries + os.pathsep + os.environ.get("PATH", "")
            break
