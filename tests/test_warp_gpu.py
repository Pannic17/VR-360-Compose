"""The GPU warp must be indistinguishable from the CPU plan -- metric D, on the device.

Skipped as a whole when `cupy` or a CUDA device is missing; the gate that decides that is
tested without either in `tests/test_device.py`. The synthetic tiles here are small, so
this is about the arithmetic, not the speed: real 8K numbers are `tools/gpu_probe.py`'s.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from conftest import analytic_panorama, sample_panorama_into_tile
from vr_compose import device, warp_gpu
from vr_compose.rig import Rig, twenty_file_rig
from vr_compose.stitch import BIT_DEPTHS, SAMPLERS, stitch_frame
from vr_compose.warp import WarpPlan
from vr_compose.warp_gpu import GpuUnavailable, GpuWarpPlan

U8 = npt.NDArray[np.uint8]
WIDTH, TILE = 512, 96

_found = device.probe_cuda()
needs_gpu = pytest.mark.skipif(
    not (_found.importable and not _found.error and _found.count > 0),
    reason=f"no usable CUDA device: {_found.error or 'none found'}",
)


@pytest.fixture(scope="module")
def rig() -> Rig:
    return twenty_file_rig()


@pytest.fixture(scope="module")
def tiles(rig: Rig) -> dict[int, U8]:
    panorama = analytic_panorama(WIDTH, WIDTH // 2)
    return {i: sample_panorama_into_tile(panorama, rig, i, TILE) for i in rig.unique_indices}


@pytest.fixture(scope="module", params=list(SAMPLERS))
def plans(rig: Rig, request: pytest.FixtureRequest) -> tuple[WarpPlan, GpuWarpPlan]:
    if not (_found.importable and not _found.error and _found.count > 0):
        pytest.skip("no usable CUDA device")
    plan = WarpPlan.build(rig, WIDTH, WIDTH // 2, TILE, sampler=request.param)
    return plan, GpuWarpPlan.from_plan(plan)


@needs_gpu
@pytest.mark.parametrize("bit_depth", list(BIT_DEPTHS))
@pytest.mark.parametrize("with_stats", [False, True])
def test_gpu_matches_the_cpu_plan_byte_for_byte(
    plans: tuple[WarpPlan, GpuWarpPlan], tiles: dict[int, U8], bit_depth: int, with_stats: bool
) -> None:
    """Images and the gate's statistics, per sampler, per depth, with and without stats."""
    plan, gpu = plans
    expected = plan.apply(tiles, with_stats=with_stats, bit_depth=bit_depth)
    got = gpu.apply(tiles, with_stats=with_stats, bit_depth=bit_depth)
    assert got.image.dtype == expected.image.dtype and got.image.shape == expected.image.shape
    assert np.array_equal(got.image, expected.image), (
        f"{plan.sampler} {bit_depth}-bit: {np.count_nonzero(got.image != expected.image)} differ"
    )
    assert np.array_equal(got.stats.count, expected.stats.count)
    assert np.array_equal(got.stats.luma_sum, expected.stats.luma_sum)
    assert np.array_equal(got.stats.luma_sq_sum, expected.stats.luma_sq_sum)


@needs_gpu
def test_gpu_matches_the_per_frame_stitch(
    plans: tuple[WarpPlan, GpuWarpPlan], rig: Rig, tiles: dict[int, U8]
) -> None:
    """Transitively true, but P1's stitch is the reference, so say it directly."""
    plan, gpu = plans
    expected = stitch_frame(tiles, rig, WIDTH, sampler=plan.sampler).image
    assert np.array_equal(gpu.apply(tiles).image, expected)


@needs_gpu
def test_a_second_frame_reuses_the_device_buffers(
    plans: tuple[WarpPlan, GpuWarpPlan], tiles: dict[int, U8]
) -> None:
    """The accumulators are zeroed between frames and the tile buffers are overwritten,
    so frame two must not remember frame one."""
    plan, gpu = plans
    first = gpu.apply(tiles).image
    darker = {camera: (tile // 2).astype(np.uint8) for camera, tile in tiles.items()}
    second = gpu.apply(darker).image
    assert np.array_equal(second, plan.apply(darker).image)
    assert not np.array_equal(first, second)
    assert np.array_equal(gpu.apply(tiles).image, first)


@needs_gpu
def test_the_gpu_plan_stands_in_for_the_cpu_plan(plans: tuple[WarpPlan, GpuWarpPlan]) -> None:
    plan, gpu = plans
    for name in ("width", "height", "tile_size", "sampler", "feather_power", "fingerprint"):
        assert getattr(gpu, name) == getattr(plan, name), name
    assert gpu.build_seconds == plan.build_seconds and gpu.nbytes == plan.nbytes
    assert gpu.device_bytes > plan.nbytes  # the plan, plus buffers, plus (cubic) weights
    assert gpu.upload_seconds >= 0.0


@needs_gpu
def test_gpu_apply_validates_like_the_cpu(
    plans: tuple[WarpPlan, GpuWarpPlan], tiles: dict[int, U8]
) -> None:
    _plan, gpu = plans
    with pytest.raises(ValueError, match="bit_depth must be one of"):
        gpu.apply(tiles, bit_depth=12)
    missing = dict(tiles)
    del missing[next(iter(missing))]
    with pytest.raises(ValueError, match="missing tile"):
        gpu.apply(missing)


def test_the_fmad_switch_is_off() -> None:
    """Point 1 of the module docstring; the byte equality above depends on it."""
    assert "--fmad=false" in warp_gpu.NVRTC_OPTIONS


def test_without_cupy_from_plan_says_so(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Whatever the import raises, the caller gets one exception type to fall back on."""

    def broken() -> object:
        raise GpuUnavailable("cupy is not usable: ImportError: simulated")

    monkeypatch.setattr(warp_gpu, "_cupy", broken)
    plan = WarpPlan.build(rig, 64, 32, 16, sampler="nearest")
    with pytest.raises(GpuUnavailable, match="simulated"):
        GpuWarpPlan.from_plan(plan)
