"""The same warp on a CUDA GPU: :class:`WarpPlan`'s contributor lists resident on the device.

**This is not a second implementation of the geometry.** A :class:`GpuWarpPlan` is built
*from* a :class:`vr_compose.warp.WarpPlan` and uploads its arrays; the plan is still what
P1's `stitch_bands` verified, and the fingerprint, the on-disk cache and the pipeline's
checks are all the CPU plan's. What runs here is only the per-frame gather and blend.

It reproduces :meth:`WarpPlan.apply` **byte for byte** -- images at 8 and 16 bits and the
overlap statistics the geometry gate reads -- measured on real 8K frames for all three
samplers (`tools/gpu_probe.py`, 2026-09-09). Four things make that possible, and every
one of them is load-bearing:

1. **No fused multiply-add.** numpy rounds `a * b` and then `+ c`; the CUDA compiler
   fuses them by default and rounds once. Compiled with `--fmad=false`. Division is left
   at its IEEE default (`--prec-div=true`).
2. **The CPU's operation order**, left to right in float32, with the CPU's constants:
   :func:`stitch.bilinear_blend`, :func:`stitch.cubic_blend`, :func:`stitch.luma_bt709`
   and :func:`stitch.quantise` transcribed statement for statement.
3. **Catmull-Rom weights computed on the host** by :func:`stitch.catmull_rom_weights`
   -- the very function the CPU uses -- and uploaded with the plan. They depend on the
   plan, not the frame, so this trades 32 bytes an entry of device memory (about 2.4 GB
   at 8K) for never having to trust an in-kernel polynomial to round like numpy.
4. **One launch per tile, tiles in plan order, no atomics.** Inside one tile every output
   pixel occurs at most once (`WarpPlan.build` emits each visible direction once per
   view), so a plain read-modify-write is race-free, and across tiles the accumulation
   order is exactly the CPU's `for tile in self.tiles`. The result does not depend on
   the block size or the GPU.

Measured at 8K on an RTX 4090 (catmullrom): upload 16 ms, kernels 8.5 ms, download 16 ms
-- about 40 ms a frame against 11.0 s on 8 CPU threads. The plan and its weights hold
about 4.6 GB of device memory; the 12 GB floor in :mod:`vr_compose.device` leaves room.

`cupy` is imported lazily, so this module can be imported anywhere; only
:meth:`GpuWarpPlan.from_plan` needs it.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

import numpy as np
import numpy.typing as npt

from vr_compose.stitch import (
    _EPSILON,
    BIT_DEPTHS,
    SIXTEEN_BIT_SCALE,
    BandStats,
    Panorama,
    StitchResult,
    catmull_rom_weights,
)
from vr_compose.warp import WarpPlan

__all__ = ["NVRTC_OPTIONS", "GpuUnavailable", "GpuWarpPlan"]

U8 = npt.NDArray[np.uint8]

BLOCK = 256
NVRTC_OPTIONS = ("--fmad=false",)
"""Point 1 of the module docstring. Change this and metric D is gone."""


class GpuUnavailable(RuntimeError):
    """`cupy` (or the driver behind it) is not usable here. The caller falls back."""


def _cupy() -> Any:
    try:
        import cupy  # type: ignore[import-untyped,unused-ignore]
    except Exception as error:
        raise GpuUnavailable(f"cupy is not usable: {type(error).__name__}: {error}") from error
    return cupy


# --- kernels ----------------------------------------------------------------------------
# Comments cite the numpy statement each line transcribes. Keep them in step.

_COMMON = r"""
#define L0 0.2126f
#define L1 0.7152f
#define L2 0.0722f

__device__ __forceinline__ void accumulate(
    long long o, float r, float g, float b, float w,
    float* colour, float* weight_sum, int with_stats,
    float* count, double* luma_sum, double* luma_sq_sum)
{
    // colour[out_index] += sampled * w[:, None]      (product rounded, then the sum)
    colour[3 * o + 0] = colour[3 * o + 0] + r * w;
    colour[3 * o + 1] = colour[3 * o + 1] + g * w;
    colour[3 * o + 2] = colour[3 * o + 2] + b * w;
    // weight[out_index] += w
    weight_sum[o] = weight_sum[o] + w;
    if (with_stats) {
        // luma_bt709: rgb[:,0]*L0 + rgb[:,1]*L1 + rgb[:,2]*L2, float32, left to right
        float luma = r * L0 + g * L1 + b * L2;
        count[o] = count[o] + 1.0f;                     // stats.count[out_index] += 1.0
        luma_sum[o] = luma_sum[o] + (double)luma;       // luma_sum += luma   (f32 -> f64)
        double l = (double)luma;
        luma_sq_sum[o] = luma_sq_sum[o] + l * l;        // luma_sq_sum += f64(luma) ** 2
    }
}

// upsample_grid, statement for statement: bilinear sample of the (gh, gw, 3) correction grid
// at this output pixel, subtracted from the sample before it is weighted.
__device__ __forceinline__ void correct(
    long long o, int width, const float* __restrict__ grid, int gh, int gw, float sy, float sx,
    float* r, float* g, float* b)
{
    float y = (float)(o / width);                        // out_index // width
    float x = (float)(o % width);                        // out_index % width
    float fy = (y + 0.5f) * sy - 0.5f;                   // (y + half) * sy - half
    float fx = (x + 0.5f) * sx - 0.5f;
    float y0f = fminf(fmaxf(floorf(fy), 0.0f), (float)(gh - 2));   // np.clip(np.floor(fy), 0, gh-2)
    float x0f = fminf(fmaxf(floorf(fx), 0.0f), (float)(gw - 2));
    float ty = fminf(fmaxf(fy - y0f, 0.0f), 1.0f);        // np.clip(fy - y0f, 0, 1)
    float tx = fminf(fmaxf(fx - x0f, 0.0f), 1.0f);
    int y0 = (int)y0f, x0 = (int)x0f;
    float one_tx = 1.0f - tx, one_ty = 1.0f - ty;
    const float* g00 = grid + ((long long)y0 * gw + x0) * 3;
    const float* g01 = g00 + 3;
    const float* g10 = g00 + (long long)gw * 3;
    const float* g11 = g10 + 3;
    float* out[3] = {r, g, b};
    for (int c = 0; c < 3; ++c) {
        float top = g00[c] * one_tx + g01[c] * tx;          // grid[y0,x0]*(1-tx) + grid[y0,x0+1]*tx
        float bottom = g10[c] * one_tx + g11[c] * tx;
        float v = top * one_ty + bottom * ty;               // top*(1-ty) + bottom*ty
        *out[c] = *out[c] - v;                              // sampled - upsample_grid(...)
    }
}
"""

_NEAREST = (
    _COMMON
    + r"""
extern "C" __global__ void gather_nearest(
    const unsigned char* __restrict__ src, const int* __restrict__ out_index,
    const int* __restrict__ src_index, const float* __restrict__ weight, int n,
    float* colour, float* weight_sum, int with_stats,
    float* count, double* luma_sum, double* luma_sq_sum,
    const float* __restrict__ corr, int has_corr, int width, int gh, int gw, float sy, float sx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long o = out_index[i];
    long long s = src_index[i];
    float w = weight[i];
    // sampled = source[src_index].astype(np.float32)
    float r = (float)src[3 * s + 0];
    float g = (float)src[3 * s + 1];
    float b = (float)src[3 * s + 2];
    if (has_corr) correct(o, width, corr, gh, gw, sy, sx, &r, &g, &b);
    accumulate(o, r, g, b, w, colour, weight_sum, with_stats, count, luma_sum, luma_sq_sum);
}
"""
)

_BILINEAR = (
    _COMMON
    + r"""
extern "C" __global__ void gather_bilinear(
    const unsigned char* __restrict__ src, const int* __restrict__ out_index,
    const int* __restrict__ src_index, const float* __restrict__ weight,
    const float* __restrict__ fx, const float* __restrict__ fy, int step, int n,
    float* colour, float* weight_sum, int with_stats,
    float* count, double* luma_sum, double* luma_sq_sum,
    const float* __restrict__ corr, int has_corr, int width, int gh, int gw, float sy, float sx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long o = out_index[i];
    long long s = src_index[i];
    float w = weight[i];
    float wx = fx[i], wy = fy[i];
    float one_wx = 1.0f - wx, one_wy = 1.0f - wy;       // one - wx, one - wy
    float rgb[3];
    for (int c = 0; c < 3; ++c) {
        float tl = (float)src[3 * s + c];               // source[src_index]
        float tr = (float)src[3 * (s + 1) + c];         // source[src_index + 1]
        float bl = (float)src[3 * (s + step) + c];      // source[src_index + step]
        float br = (float)src[3 * (s + step + 1) + c];  // source[src_index + step + 1]
        float top = tl * one_wx + tr * wx;              // top_left*(one-wx) + top_right*wx
        float bottom = bl * one_wx + br * wx;           // bottom_left*(one-wx) + bottom_right*wx
        rgb[c] = top * one_wy + bottom * wy;            // top*(one-wy) + bottom*wy
    }
    if (has_corr) correct(o, width, corr, gh, gw, sy, sx, &rgb[0], &rgb[1], &rgb[2]);
    accumulate(o, rgb[0], rgb[1], rgb[2], w, colour, weight_sum, with_stats,
               count, luma_sum, luma_sq_sum);
}
"""
)

_CUBIC = (
    _COMMON
    + r"""
extern "C" __global__ void gather_cubic(
    const unsigned char* __restrict__ src, const int* __restrict__ out_index,
    const int* __restrict__ src_index, const float* __restrict__ weight,
    const float* __restrict__ wx4, const float* __restrict__ wy4, int step, int n,
    float* colour, float* weight_sum, int with_stats,
    float* count, double* luma_sum, double* luma_sq_sum,
    const float* __restrict__ corr, int has_corr, int width, int gh, int gw, float sy, float sx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long o = out_index[i];
    long long s = src_index[i];
    float w = weight[i];
    float wx0 = wx4[4 * i + 0], wx1 = wx4[4 * i + 1], wx2 = wx4[4 * i + 2], wx3 = wx4[4 * i + 3];
    float wy0 = wy4[4 * i + 0], wy1 = wy4[4 * i + 1], wy2 = wy4[4 * i + 2], wy3 = wy4[4 * i + 3];
    float rgb[3];
    for (int c = 0; c < 3; ++c) {
        float collapsed[4];
        for (int row = 0; row < 4; ++row) {
            long long start = s + (long long)row * step;   // start_of_row = base + offset*step
            float t0 = (float)src[3 * start + c];
            float t1 = (float)src[3 * (start + 1) + c];
            float t2 = (float)src[3 * (start + 2) + c];
            float t3 = (float)src[3 * (start + 3) + c];
            // cubic_blend: row[0]*wx[0] + row[1]*wx[1] + row[2]*wx[2] + row[3]*wx[3]
            collapsed[row] = t0 * wx0 + t1 * wx1 + t2 * wx2 + t3 * wx3;
        }
        // collapsed[0]*wy[0] + collapsed[1]*wy[1] + collapsed[2]*wy[2] + collapsed[3]*wy[3]
        rgb[c] = collapsed[0] * wy0 + collapsed[1] * wy1 + collapsed[2] * wy2 + collapsed[3] * wy3;
    }
    if (has_corr) correct(o, width, corr, gh, gw, sy, sx, &rgb[0], &rgb[1], &rgb[2]);
    accumulate(o, rgb[0], rgb[1], rgb[2], w, colour, weight_sum, with_stats,
               count, luma_sum, luma_sq_sum);
}
"""
)

_FINISH = r"""
// blended = colour / np.maximum(weight, _EPSILON)[:, None]; quantise(blended, bit_depth)
extern "C" __global__ void finish8(
    const float* __restrict__ colour, const float* __restrict__ weight_sum,
    float eps, long long pixels, unsigned char* out)
{
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pixels) return;
    float w = fmaxf(weight_sum[p], eps);
    for (int c = 0; c < 3; ++c) {
        float v = colour[3 * p + c] / w;
        v = fminf(fmaxf(v, 0.0f), 255.0f);              // np.clip(blended, 0.0, 255.0)
        out[3 * p + c] = (unsigned char)rintf(v);       // np.rint -> uint8
    }
}

extern "C" __global__ void finish16(
    const float* __restrict__ colour, const float* __restrict__ weight_sum,
    float eps, float scale, long long pixels, unsigned short* out)
{
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pixels) return;
    float w = fmaxf(weight_sum[p], eps);
    for (int c = 0; c < 3; ++c) {
        float v = colour[3 * p + c] / w;
        v = v * scale;                                  // blended * SIXTEEN_BIT_SCALE
        v = fminf(fmaxf(v, 0.0f), 65535.0f);            // np.clip(scaled, 0.0, 65535.0)
        out[3 * p + c] = (unsigned short)rintf(v);      // np.rint -> uint16
    }
}
"""

_GATHER = {
    "nearest": (_NEAREST, "gather_nearest"),
    "bilinear": (_BILINEAR, "gather_bilinear"),
    "catmullrom": (_CUBIC, "gather_cubic"),
}


@dataclasses.dataclass(slots=True)
class _DeviceTile:
    camera: int
    n: int
    out_index: Any
    src_index: Any
    weight: Any
    fx: Any = None
    fy: Any = None
    wx: Any = None
    wy: Any = None


class GpuWarpPlan:
    """A :class:`WarpPlan` resident on the GPU, with the same :meth:`apply`.

    Build one with :meth:`from_plan`. It keeps the CPU plan and forwards its attributes,
    so the pipeline can treat either interchangeably. `threads` is accepted by
    :meth:`apply` for signature compatibility and ignored.
    """

    def __init__(self, plan: WarpPlan, cp: Any) -> None:
        self.plan = plan
        self._cp = cp
        started = time.time()
        source, name = _GATHER[plan.sampler]
        self._gather = cp.RawKernel(source, name, options=NVRTC_OPTIONS)
        finish = cp.RawModule(code=_FINISH, options=NVRTC_OPTIONS)
        self._finish8 = finish.get_function("finish8")
        self._finish16 = finish.get_function("finish16")

        self._tiles: list[_DeviceTile] = []
        for tile in plan.tiles:
            if tile.out_index.size > 1 and not bool(np.all(np.diff(tile.out_index) > 0)):
                raise ValueError(
                    f"camera {tile.camera}: output indices must be strictly increasing "
                    "within a tile (point 4 of the module docstring)"
                )
            entry = _DeviceTile(
                camera=tile.camera,
                n=int(tile.out_index.size),
                out_index=cp.asarray(tile.out_index),
                src_index=cp.asarray(tile.src_index),
                weight=cp.asarray(tile.weight),
            )
            if plan.sampler == "bilinear":
                entry.fx = cp.asarray(tile.fx)
                entry.fy = cp.asarray(tile.fy)
            elif plan.sampler == "catmullrom":
                entry.wx = cp.asarray(_stacked_weights(tile.fx))
                entry.wy = cp.asarray(_stacked_weights(tile.fy))
            self._tiles.append(entry)

        pixels = plan.width * plan.height
        self._pixels = pixels
        self._colour = cp.zeros(pixels * 3, cp.float32)
        self._weight = cp.zeros(pixels, cp.float32)
        self._count = cp.zeros(pixels, cp.float32)
        self._luma_sum = cp.zeros(pixels, cp.float64)
        self._luma_sq_sum = cp.zeros(pixels, cp.float64)
        self._out8 = cp.empty(pixels * 3, cp.uint8)
        self._out16 = cp.empty(pixels * 3, cp.uint16)
        self._sources: dict[int, Any] = {}
        self._no_corr = cp.zeros(12, cp.float32)  # a valid pointer for the has_corr=0 launches
        cp.cuda.Device().synchronize()
        self.upload_seconds = time.time() - started

    @classmethod
    def from_plan(cls, plan: WarpPlan) -> GpuWarpPlan:
        """Upload `plan`. Raises :class:`GpuUnavailable` when `cupy` cannot be used."""
        return cls(plan, _cupy())

    # -- forwarded so a GpuWarpPlan can stand where a WarpPlan is expected --------------

    @property
    def width(self) -> int:
        return self.plan.width

    @property
    def height(self) -> int:
        return self.plan.height

    @property
    def tile_size(self) -> int:
        return self.plan.tile_size

    @property
    def sampler(self) -> str:
        return self.plan.sampler

    @property
    def feather_power(self) -> float:
        return self.plan.feather_power

    @property
    def fingerprint(self) -> str:
        return self.plan.fingerprint

    @property
    def build_seconds(self) -> float:
        return self.plan.build_seconds

    @property
    def nbytes(self) -> int:
        return self.plan.nbytes

    @property
    def device_bytes(self) -> int:
        """Device memory held by the plan, its weights and the working buffers."""
        return int(self._cp.get_default_memory_pool().used_bytes())

    # -- the per-frame work --------------------------------------------------------------

    def apply(
        self,
        tiles: dict[int, U8],
        *,
        with_stats: bool = False,
        threads: int = 0,
        bit_depth: int = 8,
        corrections: dict[int, npt.NDArray[np.float32]] | None = None,
    ) -> StitchResult:
        """Blend one frame's tiles. Same result as :meth:`WarpPlan.apply`, bit for bit.

        `corrections` are the harmonisation grids (camera -> `(gh, gw, 3)` float32); each
        is uploaded (a few hundred KB) and applied in-kernel by `correct()`, which is
        :func:`vr_compose.warp.upsample_grid` transcribed.
        """
        del threads  # the GPU has its own idea of parallelism
        self.plan._validate(tiles)
        if bit_depth not in BIT_DEPTHS:
            raise ValueError(f"bit_depth must be one of {BIT_DEPTHS}, got {bit_depth}")
        cp = self._cp
        self._upload(tiles)

        self._colour.fill(0)
        self._weight.fill(0)
        if with_stats:
            self._count.fill(0)
            self._luma_sum.fill(0)
            self._luma_sq_sum.fill(0)
        step = np.int32(self.plan.tile_size)
        flag = np.int32(1 if with_stats else 0)
        stats_args = (
            self._colour,
            self._weight,
            flag,
            self._count,
            self._luma_sum,
            self._luma_sq_sum,
        )
        width = np.int32(self.plan.width)
        if corrections is not None:
            first = next(iter(corrections.values()))
            gh, gw = int(first.shape[0]), int(first.shape[1])
            sy = np.float32(gh / self.plan.height)
            sx = np.float32(gw / self.plan.width)
            corr_shape = (np.int32(1), width, np.int32(gh), np.int32(gw), sy, sx)
        else:
            corr_shape = (
                np.int32(0),
                width,
                np.int32(2),
                np.int32(2),
                np.float32(0),
                np.float32(0),
            )
        for entry in self._tiles:
            if entry.n == 0:
                continue
            grid = ((entry.n + BLOCK - 1) // BLOCK,)
            head = (self._sources[entry.camera], entry.out_index, entry.src_index, entry.weight)
            n = np.int32(entry.n)
            if corrections is not None:
                block = np.ascontiguousarray(corrections[entry.camera], dtype=np.float32)
                if block.shape != (gh, gw, 3):
                    raise ValueError(f"camera {entry.camera}: correction grid {block.shape}")
                corr = cp.asarray(block)
            else:
                corr = self._no_corr
            tail = (*stats_args, corr, *corr_shape)
            if self.plan.sampler == "nearest":
                self._gather(grid, (BLOCK,), (*head, n, *tail))
            elif self.plan.sampler == "bilinear":
                self._gather(grid, (BLOCK,), (*head, entry.fx, entry.fy, step, n, *tail))
            else:
                self._gather(grid, (BLOCK,), (*head, entry.wx, entry.wy, step, n, *tail))

        grid = ((self._pixels + BLOCK - 1) // BLOCK,)
        pixels = np.int64(self._pixels)
        eps = np.float32(_EPSILON)
        if bit_depth == 8:
            self._finish8(grid, (BLOCK,), (self._colour, self._weight, eps, pixels, self._out8))
            image: Panorama = cp.asnumpy(self._out8)
        else:
            scale = np.float32(SIXTEEN_BIT_SCALE)
            self._finish16(
                grid, (BLOCK,), (self._colour, self._weight, eps, scale, pixels, self._out16)
            )
            image = cp.asnumpy(self._out16)

        if with_stats:
            stats = BandStats(
                count=cp.asnumpy(self._count),
                luma_sum=cp.asnumpy(self._luma_sum),
                luma_sq_sum=cp.asnumpy(self._luma_sq_sum),
            )
        else:
            stats = BandStats.zeros(0)
        return StitchResult(image=image.reshape(self.plan.height, self.plan.width, 3), stats=stats)

    def _upload(self, tiles: dict[int, U8]) -> None:
        for entry in self._tiles:
            host = np.ascontiguousarray(tiles[entry.camera]).reshape(-1)
            buffer = self._sources.get(entry.camera)
            if buffer is None:
                self._sources[entry.camera] = self._cp.asarray(host)
            else:
                buffer.set(host)


def _stacked_weights(fraction: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """The four Catmull-Rom weights per entry, interleaved, from the CPU's own function."""
    if fraction.size == 0:
        return np.zeros((0, 4), np.float32)
    return np.ascontiguousarray(np.stack(catmull_rom_weights(fraction), axis=1), dtype=np.float32)
