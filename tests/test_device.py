"""The GPU gate: `cpu` always, `cuda` only when the machine qualifies, otherwise a warning.

None of this needs a GPU, or `cupy`. The probe is injected, so every branch of the rule
set the user gave on 2026-09-09 is exercised on any machine.
"""

from __future__ import annotations

import pytest

from vr_compose import device
from vr_compose.device import CudaProbe, resolve_device

GB = 10**9
MIB = 2**20


def probe_with(**fields: object) -> CudaProbe:
    defaults: dict[str, object] = dict(
        importable=True, count=1, name="Test GPU", total_bytes=24 * GB
    )
    defaults.update(fields)
    return CudaProbe(**defaults)  # type: ignore[arg-type]


def test_cpu_is_always_honoured_without_probing() -> None:
    def never() -> CudaProbe:
        raise AssertionError("cpu must not probe the GPU")

    placed = resolve_device("cpu", probe=never)
    assert placed.device == "cpu" and placed.warning is None and not placed.fell_back


def test_an_unknown_device_is_refused() -> None:
    with pytest.raises(ValueError, match="device must be one of"):
        resolve_device("tpu")


def test_a_qualifying_gpu_is_used_and_named() -> None:
    placed = resolve_device("cuda", probe=lambda: probe_with(name="NVIDIA GeForce RTX 4090"))
    assert placed.device == "cuda" and placed.warning is None
    assert "RTX 4090" in placed.detail and "24.0 GB" in placed.detail


def test_auto_takes_a_qualifying_gpu() -> None:
    placed = resolve_device("auto", probe=lambda: probe_with(name="NVIDIA GeForce RTX 4090"))
    assert placed.device == "cuda" and placed.warning is None and placed.note == ""


@pytest.mark.parametrize(
    "found",
    [
        CudaProbe(importable=False, error="ModuleNotFoundError: cupy"),
        CudaProbe(importable=True, error="CUDARuntimeError: no driver"),
        CudaProbe(importable=True, count=0),
        CudaProbe(importable=True, count=1, name="Small GPU", total_bytes=8 * GB),
    ],
)
def test_auto_takes_the_cpu_quietly_when_the_gate_says_no(found: CudaProbe) -> None:
    """Rule 4: the GUI asked for nothing in particular, so this is a note, not a warning."""
    placed = resolve_device("auto", probe=lambda: found)
    assert placed.device == "cpu"
    assert placed.warning is None and not placed.fell_back
    assert placed.note and placed.note.endswith("; using the CPU")
    # the same reason a `cuda` request would have been warned with
    warned = resolve_device("cuda", probe=lambda: found)
    assert warned.warning is not None
    assert placed.note.removesuffix("; using the CPU") in warned.warning


def test_a_card_newer_than_the_compiler_is_refused_before_the_upload() -> None:
    """A 50-series card against a CUDA older than 12.8: the gate says so, rather than the
    render discovering it after building and uploading a 4.6 GB plan.

    cupy compiles for `min(card, ceiling)` and emits SASS, so a cubin for the wrong
    architecture simply will not load. The numbers are cupy's own: NVRTC 12.0-12.7 stop
    at 90, and a Blackwell card is 120.
    """
    found = probe_with(
        name="NVIDIA GeForce RTX 5090",
        total_bytes=32 * GB,
        compute_capability="120",
        nvrtc_ceiling="90",
    )
    placed = resolve_device("cuda", probe=lambda: found)
    assert placed.device == "cpu" and placed.fell_back
    assert placed.warning is not None
    assert "compute capability 120" in placed.warning
    assert "no further than 90" in placed.warning
    assert "12.8" in placed.warning


def test_the_same_card_passes_once_the_compiler_can_reach_it() -> None:
    found = probe_with(
        name="NVIDIA GeForce RTX 5090",
        total_bytes=32 * GB,
        compute_capability="120",
        nvrtc_ceiling="121",
    )
    assert resolve_device("cuda", probe=lambda: found).device == "cuda"


@pytest.mark.parametrize("fields", [{"nvrtc_ceiling": ""}, {"compute_capability": ""},
                                    {"compute_capability": "sm_120"}])  # fmt: skip
def test_an_unreadable_pair_costs_the_check_and_not_the_gpu(fields: dict[str, str]) -> None:
    """`probe_cuda` reads both through `_read`, and cupy could rename what it reads from.

    When that happens the gate must lose its opinion, not the card: the render still has
    its own fallback if the compile turns out to fail.
    """
    found = probe_with(**{"compute_capability": "120", "nvrtc_ceiling": "90", **fields})
    assert resolve_device("cuda", probe=lambda: found).device == "cuda"


def test_missing_cupy_warns_and_falls_back() -> None:
    placed = resolve_device(
        "cuda", probe=lambda: CudaProbe(importable=False, error="ModuleNotFoundError: cupy")
    )
    assert placed.device == "cpu" and placed.fell_back
    assert placed.warning is not None
    assert "vr-compose[gpu]" in placed.warning and "falling back to the CPU" in placed.warning
    assert "ModuleNotFoundError" in placed.warning


def test_no_device_warns_and_falls_back() -> None:
    placed = resolve_device("cuda", probe=lambda: probe_with(count=0, name="", total_bytes=0))
    assert placed.device == "cpu"
    assert placed.warning is not None and "no CUDA device" in placed.warning


def test_a_runtime_error_warns_and_falls_back() -> None:
    placed = resolve_device(
        "cuda", probe=lambda: CudaProbe(importable=True, error="CUDARuntimeError: no driver")
    )
    assert placed.device == "cpu"
    assert placed.warning is not None and "could not be initialised" in placed.warning
    assert "no driver" in placed.warning


@pytest.mark.parametrize(
    ("total_bytes", "qualifies"),
    [
        (8 * GB, False),  # an 8 GB card
        (11264 * MIB, False),  # an 11 GB card (1080 Ti class)
        (12282 * MIB, True),  # what a "12 GB" card actually reports: just under 12 GiB
        (12 * GB, True),  # exactly the decimal floor
        (12 * GB - 1, False),  # one byte under it
        (16376 * MIB, True),  # a "16 GB" card
        (24564 * MIB, True),  # the 4090
    ],
)
def test_the_12_gb_floor_is_decimal(total_bytes: int, qualifies: bool) -> None:
    """A card sold as 12 GB reports a little under 12 GiB and must pass. The floor was 16 GB
    for a few hours on 2026-09-09; the user lowered it to 12 the same day."""
    placed = resolve_device("cuda", probe=lambda: probe_with(total_bytes=total_bytes))
    assert (placed.device == "cuda") is qualifies
    if not qualifies:
        assert placed.warning is not None
        assert "below the 12 GB floor" in placed.warning
        assert f"{total_bytes / 1e9:.1f} GB" in placed.warning


def test_the_default_probe_is_looked_up_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pipeline calls `resolve_device` without a probe; tests replace the module's."""
    monkeypatch.setattr(device, "probe_cuda", lambda: probe_with(count=0))
    assert resolve_device("cuda").device == "cpu"
    monkeypatch.setattr(device, "probe_cuda", lambda: probe_with())
    assert resolve_device("cuda").device == "cuda"


def test_the_real_probe_never_raises() -> None:
    """Whatever this machine has -- no cupy, no driver, a card -- the answer is a value."""
    found = device.probe_cuda()
    assert isinstance(found, CudaProbe)
    if found.importable and not found.error and found.count:
        assert found.name and found.total_bytes > 0
