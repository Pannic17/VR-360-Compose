"""Where the warp runs: the CPU always, a CUDA GPU only when asked for and only when it fits.

The CPU path is the byte-exact reference implementation (AGENTS.md section 9, metric D);
the GPU path in :mod:`vr_compose.warp_gpu` reproduces it bit for bit but exists on one
kind of hardware. Four rules, set by the user on 2026-09-09:

1. The library and CLI default is `cpu`. `cuda` asks for the GPU explicitly.
2. Asking for `cuda` on a machine that cannot serve it is a **warning, not an error**:
   the job falls back to the CPU and runs. No NVIDIA driver, no `cupy`, no device, or a
   device with less than 12 GB of memory all take that path (16 GB until the user lowered
   it, same day).
3. The 12 GB floor is measured in decimal gigabytes. A card sold as "12 GB" reports
   about 12 282 MiB (a little under 12 GiB), and it must pass; an 8 or 11 GB card must
   not. The plan and its weights need about 4.6 GB at 8K, so 12 GB leaves room.
4. `auto` -- what the GUI sends -- takes the GPU when the same gate passes and the CPU
   otherwise, **quietly**: nothing was asked for that failed, so the reason goes into the
   log as a note, not into the run's warnings. The GUI has no switch; it just gets the
   fastest device the machine has.

The probe is injectable so the gate can be tested without a GPU, and without `cupy`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

__all__ = [
    "DEFAULT_DEVICE",
    "DEVICES",
    "GUI_DEVICE",
    "MIN_CUDA_MEMORY_BYTES",
    "CudaProbe",
    "Placement",
    "probe_cuda",
    "resolve_device",
]

DEVICES = ("cpu", "cuda", "auto")
DEFAULT_DEVICE = "cpu"
GUI_DEVICE = "auto"
"""What the GUI asks for (rule 4). The GUI has no device control; this is the policy."""
MIN_CUDA_MEMORY_BYTES = 12 * 10**9
"""Decimal on purpose; see rule 3 in the module docstring."""


@dataclasses.dataclass(frozen=True, slots=True)
class CudaProbe:
    """What one look at the machine found. `error` is set when `cupy` itself failed."""

    importable: bool
    error: str = ""
    count: int = 0
    name: str = ""
    total_bytes: int = 0

    @property
    def describe(self) -> str:
        return f"{self.name}, {self.total_bytes / 1e9:.1f} GB"


@dataclasses.dataclass(frozen=True, slots=True)
class Placement:
    """The decision: which device the warp will actually run on, and why if not asked."""

    device: str
    """`cpu` or `cuda` -- never `auto`; that has been decided by now."""
    warning: str | None = None
    """Set when `cuda` was requested and the CPU is used instead. Advisory."""
    detail: str = ""
    """For the log: the GPU's name and memory when `device` is `cuda`."""
    note: str = ""
    """Set when `auto` chose the CPU: why, for the log. Not a warning (rule 4)."""

    @property
    def fell_back(self) -> bool:
        return self.warning is not None


def probe_cuda() -> CudaProbe:
    """Look for a usable CUDA device through `cupy`. Never raises."""
    try:
        import cupy as cp  # type: ignore[import-untyped,unused-ignore]
    except Exception as error:
        return CudaProbe(importable=False, error=f"{type(error).__name__}: {error}")
    try:
        count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as error:
        return CudaProbe(importable=True, error=f"{type(error).__name__}: {error}")
    if count == 0:
        return CudaProbe(importable=True, count=0)
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        total = int(props["totalGlobalMem"])
    except Exception as error:
        return CudaProbe(importable=True, count=count, error=f"{type(error).__name__}: {error}")
    return CudaProbe(importable=True, count=count, name=str(name), total_bytes=total)


def resolve_device(
    requested: str,
    *,
    probe: Callable[[], CudaProbe] | None = None,
    minimum_bytes: int | None = None,
) -> Placement:
    """Decide where the warp runs. `cpu` is always honoured; `cuda` has to earn it.

    `probe` and `minimum_bytes` default to :func:`probe_cuda` and
    :data:`MIN_CUDA_MEMORY_BYTES`, looked up at call time so a test can replace the module
    attributes and exercise the pipeline's fallback -- or its GPU path -- on any machine.
    """
    if requested not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {requested!r}")
    if requested == "cpu":
        return Placement(device="cpu")

    found = (probe or probe_cuda)()
    if minimum_bytes is None:
        minimum_bytes = MIN_CUDA_MEMORY_BYTES
    reason = _why_not(found, minimum_bytes)
    if reason is None:
        return Placement(device="cuda", detail=found.describe)
    if requested == "auto":
        return Placement(device="cpu", note=f"{reason}; using the CPU")
    return Placement(
        device="cpu",
        warning=f"GPU requested (--device cuda) but {reason}; falling back to the CPU.",
    )


def _why_not(found: CudaProbe, minimum_bytes: int) -> str | None:
    """The gate. `None` means the GPU may be used; otherwise the reason it may not."""
    if not found.importable:
        return (
            "cupy is not installed or the NVIDIA driver is missing "
            f"(pip install 'vr-compose[gpu]' on a machine with an NVIDIA card) [{found.error}]"
        )
    if found.error:
        return f"CUDA could not be initialised [{found.error}]"
    if found.count == 0:
        return "no CUDA device was found"
    if found.total_bytes < minimum_bytes:
        return (
            f"{found.name} has {found.total_bytes / 1e9:.1f} GB of memory, below the "
            f"{minimum_bytes / 1e9:.0f} GB floor"
        )
    return None
