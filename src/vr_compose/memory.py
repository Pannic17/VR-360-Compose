"""Peak memory of a job, for the 64 GB soft cap.

The cap is **advisory**: a job that exceeds it warns and keeps running. That is a
deliberate decision (ROADMAP P3 item 0) -- an 8K run needs about 18 GB and a 16K master
about 27 GB, both far inside the cap on the machines this ships to, so the number exists
to catch a regression, not to gate the work.

Nothing here polls. Windows already tracks each process's high-water mark
(`PeakWorkingSetSize`), so one query per process gives that process's true peak no matter
when it is asked -- as long as the process is still alive, which is why
:class:`vr_compose.encode.SegmentWriter` reads it before shutting ffmpeg down.

The reported total is the **sum of per-process peaks**, which is an upper bound on the
peak of the sum: the stitcher and the encoder do reach their high-water marks at
overlapping times, but not necessarily the same instant. Over-reporting is the right
direction for a warning threshold.
"""

from __future__ import annotations

import ctypes
import dataclasses
import sys

__all__ = ["SOFT_LIMIT_BYTES", "PeakMemory", "peak_working_set"]

SOFT_LIMIT_BYTES = 64 * 2**30
"""Warn above this; never fail, never interrupt."""

_PROCESS_QUERY_INFORMATION = 0x0400
_PROCESS_VM_READ = 0x0010


class _MemoryCounters(ctypes.Structure):
    """PROCESS_MEMORY_COUNTERS. Only `PeakWorkingSetSize` is read; the rest pad it out."""

    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


if sys.platform == "win32":
    import ctypes.wintypes as _wintypes

    # The prototypes have to be declared. ctypes defaults an undeclared return value to
    # C `int`, which truncates a 64-bit HANDLE -- and GetCurrentProcess returns the
    # pseudo-handle (HANDLE)-1, so leaving it undeclared makes every reading of *this*
    # process come back zero while child processes, whose handles are small, still work.
    _kernel32 = ctypes.windll.kernel32
    _psapi = ctypes.windll.psapi
    _kernel32.GetCurrentProcess.restype = _wintypes.HANDLE
    _kernel32.GetCurrentProcess.argtypes = []
    _kernel32.OpenProcess.restype = _wintypes.HANDLE
    _kernel32.OpenProcess.argtypes = [_wintypes.DWORD, _wintypes.BOOL, _wintypes.DWORD]
    _kernel32.CloseHandle.restype = _wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [_wintypes.HANDLE]
    _psapi.GetProcessMemoryInfo.restype = _wintypes.BOOL
    _psapi.GetProcessMemoryInfo.argtypes = [
        _wintypes.HANDLE,
        ctypes.POINTER(_MemoryCounters),
        _wintypes.DWORD,
    ]


def _peaks(pid: int | None) -> tuple[int, int]:
    """`(peak working set, peak commit)` of `pid`, or of this process when None.

    Returns zeros rather than raising when the figures cannot be had -- the process has
    exited, the platform is not Windows, the handle is denied. A memory *report* must
    never be the thing that breaks a render.
    """
    if sys.platform != "win32":
        return 0, 0
    handle = (
        _kernel32.GetCurrentProcess()
        if pid is None
        else _kernel32.OpenProcess(_PROCESS_QUERY_INFORMATION | _PROCESS_VM_READ, False, pid)
    )
    if not handle:
        return 0, 0
    try:
        counters = _MemoryCounters()
        counters.cb = ctypes.sizeof(_MemoryCounters)
        if not _psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return 0, 0
        return int(counters.PeakWorkingSetSize), int(counters.PeakPagefileUsage)
    finally:
        if pid is not None:
            _kernel32.CloseHandle(handle)


def peak_working_set(pid: int | None = None) -> int:
    """Peak *resident* bytes: how much of the process was in RAM at its high-water mark.

    This is the honest answer to "how much memory did the run use", but it is a property
    of the machine as much as of the process -- Windows trims working sets under cache
    pressure, and the same 8K encode measured 14.8 GiB on an idle machine and 8.6 GiB on
    a busy one. Use :func:`peak_commit` when comparing configurations.
    """
    return _peaks(pid)[0]


def peak_commit(pid: int | None = None) -> int:
    """Peak *committed* bytes: how much the process asked the OS to back.

    Allocation rather than residency, so it barely moves with machine state, which makes
    it the number to compare encoder settings by.
    """
    return _peaks(pid)[1]


@dataclasses.dataclass(frozen=True, slots=True)
class PeakMemory:
    """What a job's two halves each peaked at, in commit and in residency.

    The cap is compared against **commit**, because that is what a machine actually runs
    out of: committed pages count against the system commit limit whether or not they are
    resident. Residency is reported alongside because it is what Task Manager shows, and
    the two differ by a lot -- an 8K x264 segment commits 21 GiB and keeps about 6 GiB of
    it resident.
    """

    stitcher: int = 0
    """This process's peak commit: the warp plan, frame buffers and decoded tiles."""
    encoder: int = 0
    """Peak commit of the largest segment's ffmpeg (segments run one at a time)."""
    stitcher_resident: int = 0
    encoder_resident: int = 0

    @property
    def total(self) -> int:
        return self.stitcher + self.encoder

    @property
    def total_resident(self) -> int:
        return self.stitcher_resident + self.encoder_resident

    @property
    def measured(self) -> bool:
        return self.total > 0

    def exceeds(self, limit: int = SOFT_LIMIT_BYTES) -> bool:
        return self.total > limit

    def report(self) -> str:
        if not self.measured:
            return "peak memory: not available on this platform"
        gib = 2**30
        return (
            f"peak memory: {self.total / gib:.1f} GiB committed "
            f"(stitcher {self.stitcher / gib:.1f} + encoder {self.encoder / gib:.1f}), "
            f"{self.total_resident / gib:.1f} GiB resident"
        )

    def warning(self, limit: int = SOFT_LIMIT_BYTES) -> str:
        """The soft-cap message. Advisory: the caller logs it and carries on."""
        return (
            f"peak memory {self.total / 2**30:.1f} GiB committed is above the "
            f"{limit / 2**30:.0f} GiB soft limit "
            f"(stitcher {self.stitcher / 2**30:.1f}, encoder {self.encoder / 2**30:.1f}). "
            "The run is not affected. To lower the encoder's share, pass --encoder-threads."
        )
