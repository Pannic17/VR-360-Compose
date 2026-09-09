"""Is a GPU warp worth having, and can it match the CPU byte for byte? Measure before deciding.

ROADMAP's "optional GPU acceleration" gate: no dependency enters the product until the 8K
warp has been timed on this machine *and* compared bit for bit against `WarpPlan.apply`.
This script is that measurement. It is also the prototype of `vr_compose.warp_gpu`: the
kernels here are written the way the product version must be written, because the byte
equality is a property of the arithmetic order, not of the library.

What is held fixed to make byte equality possible at all:

- **No fused multiply-add.** numpy evaluates `a * b + c` as two rounded operations; the
  CUDA compiler fuses them into one by default, which rounds differently. Compiled with
  `--fmad=false`.
- **The same operation order as `stitch.bilinear_blend` / `cubic_blend` / `luma_bt709` /
  `quantise`.** Left to right, float32, the same constants.
- **Catmull-Rom weights precomputed on the host** with the very function the CPU path
  uses, then uploaded once. They depend only on the plan, not on the frame, so this costs
  memory (32 bytes an entry, ~2.4 GB at 8K) instead of trust in in-kernel polynomial
  rounding.
- **One launch per tile, tiles in plan order, no atomics.** Within one tile every output
  pixel appears at most once, so a plain read-modify-write is race-free, and across tiles
  the accumulation order is exactly the CPU's `for tile in self.tiles`.

    python tools/gpu_probe.py                       # 8K, all three samplers, one frame
    python tools/gpu_probe.py --samplers catmullrom --iterations 10
    python tools/gpu_probe.py --frames 1656 1700 2000   # several real frames, metric D each

Read-only on the source directory. `--work` caches the warp plans between runs.
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

import numpy as np
import numpy.typing as npt

from vr_compose import io, source
from vr_compose.rig import rig_for
from vr_compose.stitch import (
    _EPSILON,
    SAMPLERS,
    SIXTEEN_BIT_SCALE,
    StitchResult,
    catmull_rom_weights,
)
from vr_compose.warp import WarpPlan, plan_fingerprint

try:
    import cupy as cp
except ImportError:  # pragma: no cover - the whole point of the probe is to have it
    print("cupy is not installed: pip install cupy-cuda12x", file=sys.stderr)
    raise SystemExit(2) from None

U8 = npt.NDArray[np.uint8]

DEFAULT_ROOT = pathlib.Path("E:/22")
BLOCK = 256

_COMMON = r"""
#define L0 0.2126f
#define L1 0.7152f
#define L2 0.0722f

__device__ __forceinline__ void accumulate(
    long long o, float r, float g, float b, float w,
    float* colour, float* weight_sum, int with_stats,
    float* count, double* luma_sum, double* luma_sq_sum)
{
    // colour[out_index] += sampled * w   (two rounded operations, no FMA)
    colour[3 * o + 0] = colour[3 * o + 0] + r * w;
    colour[3 * o + 1] = colour[3 * o + 1] + g * w;
    colour[3 * o + 2] = colour[3 * o + 2] + b * w;
    weight_sum[o] = weight_sum[o] + w;
    if (with_stats) {
        // luma_bt709: ((r*L0) + (g*L1)) + (b*L2), float32 left to right
        float luma = r * L0 + g * L1 + b * L2;
        count[o] = count[o] + 1.0f;
        luma_sum[o] = luma_sum[o] + (double)luma;
        double l = (double)luma;
        luma_sq_sum[o] = luma_sq_sum[o] + l * l;
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
    float* count, double* luma_sum, double* luma_sq_sum)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long o = out_index[i];
    long long s = src_index[i];
    float w = weight[i];
    float r = (float)src[3 * s + 0];
    float g = (float)src[3 * s + 1];
    float b = (float)src[3 * s + 2];
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
    float* count, double* luma_sum, double* luma_sq_sum)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long o = out_index[i];
    long long s = src_index[i];
    float w = weight[i];
    float wx = fx[i], wy = fy[i];
    float one_wx = 1.0f - wx, one_wy = 1.0f - wy;
    float rgb[3];
    for (int c = 0; c < 3; ++c) {
        float tl = (float)src[3 * s + c];
        float tr = (float)src[3 * (s + 1) + c];
        float bl = (float)src[3 * (s + step) + c];
        float br = (float)src[3 * (s + step + 1) + c];
        // bilinear_blend: top = tl*(1-wx) + tr*wx ; bottom likewise ; top*(1-wy) + bottom*wy
        float top = tl * one_wx + tr * wx;
        float bottom = bl * one_wx + br * wx;
        rgb[c] = top * one_wy + bottom * wy;
    }
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
    float* count, double* luma_sum, double* luma_sq_sum)
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
            long long start = s + (long long)row * step;
            float t0 = (float)src[3 * start + c];
            float t1 = (float)src[3 * (start + 1) + c];
            float t2 = (float)src[3 * (start + 2) + c];
            float t3 = (float)src[3 * (start + 3) + c];
            // cubic_blend: ((t0*wx0 + t1*wx1) + t2*wx2) + t3*wx3
            collapsed[row] = t0 * wx0 + t1 * wx1 + t2 * wx2 + t3 * wx3;
        }
        rgb[c] = collapsed[0] * wy0 + collapsed[1] * wy1 + collapsed[2] * wy2 + collapsed[3] * wy3;
    }
    accumulate(o, rgb[0], rgb[1], rgb[2], w, colour, weight_sum, with_stats,
               count, luma_sum, luma_sq_sum);
}
"""
)

_FINISH = r"""
extern "C" __global__ void finish8(
    const float* __restrict__ colour, const float* __restrict__ weight_sum,
    float eps, long long pixels, unsigned char* out)
{
    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pixels) return;
    float w = fmaxf(weight_sum[p], eps);
    for (int c = 0; c < 3; ++c) {
        float v = colour[3 * p + c] / w;
        v = fminf(fmaxf(v, 0.0f), 255.0f);
        out[3 * p + c] = (unsigned char)rintf(v);
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
        v = v * scale;
        v = fminf(fmaxf(v, 0.0f), 65535.0f);
        out[3 * p + c] = (unsigned short)rintf(v);
    }
}
"""

OPTIONS = ("--fmad=false",)


class GpuPlan:
    """The plan's contributor lists resident on the device, plus the kernels to apply them."""

    def __init__(self, plan: WarpPlan) -> None:
        self.plan = plan
        self.pixels = plan.width * plan.height
        src = {"nearest": _NEAREST, "bilinear": _BILINEAR, "catmullrom": _CUBIC}[plan.sampler]
        name = {"nearest": "gather_nearest", "bilinear": "gather_bilinear",
                "catmullrom": "gather_cubic"}[plan.sampler]  # fmt: skip
        self.gather = cp.RawKernel(src, name, options=OPTIONS)
        finish = cp.RawModule(code=_FINISH, options=OPTIONS)
        self.finish8 = finish.get_function("finish8")
        self.finish16 = finish.get_function("finish16")
        self.tiles = []
        for tile in plan.tiles:
            if tile.out_index.size and not np.all(np.diff(tile.out_index) > 0):
                raise AssertionError("out_index must be strictly increasing within a tile")
            entry = {
                "camera": tile.camera,
                "n": int(tile.out_index.size),
                "out_index": cp.asarray(tile.out_index),
                "src_index": cp.asarray(tile.src_index),
                "weight": cp.asarray(tile.weight),
            }
            if plan.sampler == "bilinear":
                entry["fx"] = cp.asarray(tile.fx)
                entry["fy"] = cp.asarray(tile.fy)
            elif plan.sampler == "catmullrom":
                # Same function as the CPU path, so the weights are the same bits.
                wx = np.stack(catmull_rom_weights(tile.fx), axis=1).astype(np.float32)
                wy = np.stack(catmull_rom_weights(tile.fy), axis=1).astype(np.float32)
                entry["wx"] = cp.asarray(np.ascontiguousarray(wx))
                entry["wy"] = cp.asarray(np.ascontiguousarray(wy))
            self.tiles.append(entry)
        self.colour = cp.zeros(self.pixels * 3, cp.float32)
        self.weight_sum = cp.zeros(self.pixels, cp.float32)
        self.count = cp.zeros(self.pixels, cp.float32)
        self.luma_sum = cp.zeros(self.pixels, cp.float64)
        self.luma_sq_sum = cp.zeros(self.pixels, cp.float64)
        self.out8 = cp.empty(self.pixels * 3, cp.uint8)
        self.out16 = cp.empty(self.pixels * 3, cp.uint16)
        self.device_tiles: dict[int, cp.ndarray] = {}

    def upload(self, tiles: dict[int, U8]) -> None:
        for entry in self.tiles:
            cam = entry["camera"]
            host = np.ascontiguousarray(tiles[cam]).reshape(-1)
            if cam in self.device_tiles:
                self.device_tiles[cam].set(host)
            else:
                self.device_tiles[cam] = cp.asarray(host)

    def run(self, *, with_stats: bool, bit_depth: int) -> None:
        self.colour.fill(0)
        self.weight_sum.fill(0)
        if with_stats:
            self.count.fill(0)
            self.luma_sum.fill(0)
            self.luma_sq_sum.fill(0)
        step = np.int32(self.plan.tile_size)
        stats_args = (
            self.count,
            self.luma_sum,
            self.luma_sq_sum,
        )
        for entry in self.tiles:
            n = entry["n"]
            if n == 0:
                continue
            grid = ((n + BLOCK - 1) // BLOCK,)
            src = self.device_tiles[entry["camera"]]
            common = (src, entry["out_index"], entry["src_index"], entry["weight"])
            tail = (self.colour, self.weight_sum, np.int32(with_stats), *stats_args)
            if self.plan.sampler == "nearest":
                self.gather(grid, (BLOCK,), (*common, np.int32(n), *tail))
            elif self.plan.sampler == "bilinear":
                self.gather(
                    grid, (BLOCK,), (*common, entry["fx"], entry["fy"], step, np.int32(n), *tail)
                )
            else:
                self.gather(
                    grid, (BLOCK,), (*common, entry["wx"], entry["wy"], step, np.int32(n), *tail)
                )
        grid = ((self.pixels + BLOCK - 1) // BLOCK,)
        if bit_depth == 8:
            args8 = (self.colour, self.weight_sum, np.float32(_EPSILON), np.int64(self.pixels))
            self.finish8(grid, (BLOCK,), (*args8, self.out8))
        else:
            self.finish16(
                grid, (BLOCK,),
                (self.colour, self.weight_sum, np.float32(_EPSILON), np.float32(SIXTEEN_BIT_SCALE),
                 np.int64(self.pixels), self.out16),
            )  # fmt: skip

    def download(self, bit_depth: int) -> np.ndarray:
        out = self.out8 if bit_depth == 8 else self.out16
        return cp.asnumpy(out).reshape(self.plan.height, self.plan.width, 3)

    def stats(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return cp.asnumpy(self.count), cp.asnumpy(self.luma_sum), cp.asnumpy(self.luma_sq_sum)

    def device_bytes(self) -> int:
        return int(cp.get_default_memory_pool().used_bytes())


def sync() -> None:
    cp.cuda.Device().synchronize()


def timed(fn) -> float:  # type: ignore[no-untyped-def]
    sync()
    started = time.perf_counter()
    fn()
    sync()
    return time.perf_counter() - started


def measure(
    gpu: GpuPlan, tiles: dict[int, U8], iterations: int
) -> tuple[float, float, float, float]:
    """Median seconds for upload, kernels, download, and kernels with statistics."""
    up = statistics.median(timed(lambda: gpu.upload(tiles)) for _ in range(iterations))
    kern = statistics.median(
        timed(lambda: gpu.run(with_stats=False, bit_depth=8)) for _ in range(iterations)
    )
    down = statistics.median(timed(lambda: gpu.download(8)) for _ in range(iterations))
    # Stats cost separately: the gate only runs every `stats_every` frames.
    with_stats = statistics.median(
        timed(lambda: gpu.run(with_stats=True, bit_depth=8)) for _ in range(3)
    )
    return up, kern, down, with_stats


def compare(label: str, a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        print(f"    {label}: SHAPE/DTYPE MISMATCH {a.shape}{a.dtype} vs {b.shape}{b.dtype}")
        return False
    if np.array_equal(a, b):
        print(f"    {label}: byte-identical")
        return True
    diff = a.astype(np.float64) - b.astype(np.float64)
    nz = np.count_nonzero(diff)
    print(
        f"    {label}: DIFFERS in {nz} of {diff.size} values "
        f"({nz / diff.size:.3%}); max |d| = {np.abs(diff).max():g}"
    )
    values, counts = np.unique(diff[diff != 0], return_counts=True)
    order = np.argsort(-counts)[:6]
    print("      histogram:", ", ".join(f"{values[i]:+g}×{counts[i]}" for i in order))
    return False


def plan_for(args: argparse.Namespace, rig, width: int, tile: int, sampler: str) -> WarpPlan:  # type: ignore[no-untyped-def]
    fingerprint = plan_fingerprint(rig, width, width // 2, tile, sampler)
    cache = (args.work / f"plan_{sampler}_{width}.npz") if args.work else None
    if cache and cache.exists():
        try:
            return WarpPlan.load(cache, fingerprint)
        except Exception as error:
            print(f"  (cache {cache.name} rejected: {error}; rebuilding)")
    started = time.perf_counter()
    plan = WarpPlan.build(rig, width, width // 2, tile, sampler=sampler)
    print(f"  plan built in {time.perf_counter() - started:.1f} s ({plan.nbytes / 2**20:.0f} MiB)")
    if cache:
        args.work.mkdir(parents=True, exist_ok=True)
        plan.save(cache)
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=pathlib.Path, default=DEFAULT_ROOT)
    parser.add_argument("--stem", default=None)
    parser.add_argument(
        "--frames", type=int, nargs="*", default=None, help="default: the first frame"
    )
    parser.add_argument("--width", type=int, default=None, help="master width; default native")
    parser.add_argument("--samplers", nargs="*", default=list(SAMPLERS), choices=list(SAMPLERS))
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--work", type=pathlib.Path, default=None, help="cache warp plans here")
    parser.add_argument(
        "--skip-cpu-timing", action="store_true", help="reuse ROADMAP's CPU numbers"
    )
    args = parser.parse_args(argv)

    sets, searched = source.discover(args.root)
    if not sets:
        raise SystemExit(f"no usable source set under {args.root} (searched {searched})")
    if args.stem:
        sets = [s for s in sets if s.stem == args.stem]
    if len(sets) != 1:
        raise SystemExit(f"choose one stem with --stem: {[s.stem for s in sets]}")
    chosen = sets[0]
    rig = rig_for(chosen.camera_count)
    tile = chosen.tile_size
    if tile is None:
        raise SystemExit("tiles are not square/uniform")
    width = args.width or rig.native_width(tile)
    frames = args.frames or [chosen.frames[0]]
    indices = list(rig.unique_indices)

    device = cp.cuda.Device(0)
    free, total = device.mem_info
    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"device     : {props['name'].decode()} {total / 1e9:.1f} GB ({free / 1e9:.1f} GB free), "
          f"cupy {cp.__version__}, runtime {cp.cuda.runtime.runtimeGetVersion()}")  # fmt: skip
    print(f"source     : {chosen.root} stem {chosen.stem!r}, {len(indices)} tiles of {tile}, "
          f"master {width}x{width // 2}, frames {frames}")  # fmt: skip
    print(f"nvrtc opts : {' '.join(OPTIONS)}")

    all_ok = True
    summary: list[tuple[str, float, float, float, float, float, bool]] = []
    for sampler in args.samplers:
        print(f"\n== {sampler} ==")
        plan = plan_for(args, rig, width, tile, sampler)
        started = time.perf_counter()
        gpu = GpuPlan(plan)
        sync()
        print(f"  uploaded plan (+weights) in {time.perf_counter() - started:.1f} s; "
              f"device pool {gpu.device_bytes() / 1e9:.2f} GB")  # fmt: skip

        cpu_seconds = float("nan")
        ok = True
        for frame_no, frame in enumerate(frames):
            tiles = io.load_tiles(chosen, frame, indices, workers=8)
            # -- CPU reference (with stats so the gate arithmetic is compared too) --
            started = time.perf_counter()
            ref8: StitchResult = plan.apply(
                tiles, with_stats=True, threads=args.cpu_threads, bit_depth=8
            )
            cpu_this = time.perf_counter() - started
            if frame_no == 0:
                cpu_seconds = cpu_this
            ref16 = plan.apply(tiles, with_stats=False, threads=args.cpu_threads, bit_depth=16)

            # -- GPU: warm-up, then timed iterations on the first frame --
            gpu.upload(tiles)
            gpu.run(with_stats=True, bit_depth=8)
            got8 = gpu.download(8)
            count, luma_sum, luma_sq = gpu.stats()
            gpu.run(with_stats=False, bit_depth=16)
            got16 = gpu.download(16)

            print(f"  frame {frame}: cpu {cpu_this:.2f} s (8-bit, with stats)")
            ok &= compare("8-bit image", got8, ref8.image)
            ok &= compare("16-bit image", got16, ref16.image)
            ok &= compare("stats.count", count, ref8.stats.count)
            ok &= compare("stats.luma_sum", luma_sum, ref8.stats.luma_sum)
            ok &= compare("stats.luma_sq_sum", luma_sq, ref8.stats.luma_sq_sum)

            if frame_no == 0:
                up, kern, down, stats_kernel = measure(gpu, tiles, args.iterations)
                total_ms = (up + kern + down) * 1e3
                print(
                    f"  gpu timing (median of {args.iterations}): upload {up * 1e3:.1f} ms, "
                    f"kernels {kern * 1e3:.1f} ms (with stats {stats_kernel * 1e3:.1f} ms), "
                    f"download {down * 1e3:.1f} ms  => {total_ms:.1f} ms/frame; "
                    f"cpu {cpu_seconds:.2f} s => {cpu_seconds * 1e3 / total_ms:.0f}x"
                )
                summary.append((sampler, cpu_seconds, up, kern, down, stats_kernel, ok))
        all_ok &= ok
        del gpu
        cp.get_default_memory_pool().free_all_blocks()

    print("\n== summary ==")
    print(
        f"{'sampler':<11}{'cpu s':>8}{'upload':>9}{'kernel':>9}{'dl':>8}"
        f"{'total ms':>10}{'speedup':>9}  metric D"
    )
    for sampler, cpu_s, up, kern, down, _stats, ok in summary:
        total_ms = (up + kern + down) * 1e3
        print(
            f"{sampler:<11}{cpu_s:>8.2f}{up * 1e3:>9.1f}{kern * 1e3:>9.1f}{down * 1e3:>8.1f}"
            f"{total_ms:>10.1f}{cpu_s * 1e3 / total_ms:>8.0f}x  "
            f"{'byte-identical' if ok else 'DIFFERS'}"
        )
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
