"""Peak-memory reporting and the 64 GB soft cap.

The cap is advisory by decision (ROADMAP P3 item 0), and "advisory" is exactly the kind
of property that regresses quietly into "fatal", so the test that matters here is that a
job which blows through the limit still succeeds.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from vr_compose import memory

WINDOWS_ONLY = pytest.mark.skipif(sys.platform != "win32", reason="Windows memory counters")


@WINDOWS_ONLY
def test_own_peaks_are_plausible() -> None:
    """Both counters read back something sane, and commit is the larger of the two.

    Commit counts pages the OS has promised to back whether or not they are resident, so
    it is >= the working set except under paging pathologies. That relationship is the
    reason the soft cap is compared against commit.
    """
    ballast = np.ones((64, 1 << 20), np.uint8)  # 64 MiB, touched so it is real
    ballast[:] = 3
    commit, resident = memory.peak_commit(), memory.peak_working_set()
    assert commit > 64 * 2**20, commit
    assert resident > 0
    assert commit >= resident * 0.5, (commit, resident)
    assert int(ballast[0, 0]) == 3  # keep it alive until after the readings


@WINDOWS_ONLY
def test_peaks_of_another_process() -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.readline()"], stdin=subprocess.PIPE
    )
    try:
        assert memory.peak_commit(child.pid) > 0
        assert memory.peak_working_set(child.pid) > 0
    finally:
        assert child.stdin is not None
        child.stdin.close()
        child.wait(timeout=30)


def test_a_dead_process_reports_zero_rather_than_raising() -> None:
    """A memory *report* must never be the thing that breaks a render."""
    unlikely_pid = 0x7FFF_FFFE
    assert memory.peak_commit(unlikely_pid) == 0
    assert memory.peak_working_set(unlikely_pid) == 0
    assert os.getpid() != unlikely_pid


def test_report_and_warning_read_clearly() -> None:
    peak = memory.PeakMemory(
        stitcher=4 * 2**30,
        encoder=21 * 2**30,
        stitcher_resident=3 * 2**30,
        encoder_resident=6 * 2**30,
    )
    assert peak.total == 25 * 2**30
    assert peak.total_resident == 9 * 2**30
    assert not peak.exceeds()  # 25 GiB is far inside the 64 GB cap
    assert peak.exceeds(limit=8 * 2**30)
    assert "25.0 GiB committed" in peak.report()
    assert "9.0 GiB resident" in peak.report()
    assert "above the 8 GiB soft limit" in peak.warning(limit=8 * 2**30)
    assert "not affected" in peak.warning(limit=8 * 2**30), "the cap must read as advisory"


def test_an_unmeasured_peak_says_so() -> None:
    blank = memory.PeakMemory()
    assert not blank.measured and not blank.exceeds()
    assert "not available" in blank.report()


def test_the_soft_limit_is_64_gib() -> None:
    assert memory.SOFT_LIMIT_BYTES == 64 * 2**30
