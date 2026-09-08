"""Precomputed warp: the fixed geometry of a rig, evaluated once, applied per frame.

Every frame of a sequence shares the same rig, output size and tile size, so everything
:mod:`vr_compose.stitch` computes per frame -- directions, projections, visibility, pixel
indices, feather weights -- is identical from frame to frame. A :class:`WarpPlan` computes
it once (about as long as one P1 stitch) and reduces each subsequent frame to a gather
and a weighted accumulate.

**The plan must reproduce :func:`vr_compose.stitch.stitch_bands` bit for bit.** It uses
the same helpers in the same order with the same dtypes, and `tests/test_warp.py` asserts
byte equality on synthetic tiles. Any future speed-up (numba, GPU, a different layout)
has to keep that property or explicitly re-baseline; P1's verified geometry is the
reference, not this module.

Layout: one contributor list per tile, sorted by output pixel. Output writes are then
monotonic and cache-friendly while the random reads stay inside a single tile, which
fits in L3. The lists total ~74 M entries at 8K (2.5 tiles per direction on average).
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import hashlib
import json
import pathlib
import time

import numpy as np
import numpy.typing as npt

from vr_compose import projection
from vr_compose.rig import Rig
from vr_compose.stitch import (
    _EPSILON,
    DEFAULT_BAND_ROWS,
    BandStats,
    StitchResult,
    _feather,
    luma_bt709,
)

F32 = npt.NDArray[np.float32]
I32 = npt.NDArray[np.int32]
U8 = npt.NDArray[np.uint8]

__all__ = ["PLAN_FORMAT", "TileContributors", "WarpPlan", "plan_fingerprint"]

PLAN_FORMAT = 1
"""Bump when the on-disk layout changes; stale caches are then rejected, not misread."""

DEFAULT_THREADS = 8
"""Measured on the 7950X at 8K. Isolated: 1 -> 3.7 s, 2 -> 2.4 s, 4 -> 1.7 s, 8 -> 1.7 s,
16 -> 2.5 s. Inside the pipeline, where the next frame's PNG decode runs concurrently and
contends for the GIL, 8 threads beat 4 (2.2 s vs 2.8 s with 8 decode workers; 1.99 s vs
2.07 s with 4). The decode pool's size matters as much -- see `SequenceJob.decode_workers`."""


def plan_fingerprint(rig: Rig, width: int, height: int, tile_size: int) -> str:
    """Identifies everything the plan depends on. Same fingerprint, same numbers."""
    payload = {
        "format": PLAN_FORMAT,
        "rig": rig.name,
        "views": [(v.yaw, v.elevation) for v in rig.views],
        "fov": rig.fov_deg,
        "mirrored": rig.mirrored,
        "width": width,
        "height": height,
        "tile": tile_size,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest[:16]


@dataclasses.dataclass(frozen=True, slots=True)
class TileContributors:
    """Where one tile lands in the output and with what weight."""

    camera: int
    out_index: I32
    """Flat output pixel index, ascending."""
    src_index: I32
    """Flat tile pixel index (row * size + column)."""
    weight: F32


@dataclasses.dataclass(frozen=True, slots=True)
class WarpPlan:
    width: int
    height: int
    tile_size: int
    fingerprint: str
    tiles: tuple[TileContributors, ...]
    build_seconds: float

    @property
    def entries(self) -> int:
        return sum(t.out_index.size for t in self.tiles)

    @property
    def nbytes(self) -> int:
        return sum(t.out_index.nbytes + t.src_index.nbytes + t.weight.nbytes for t in self.tiles)

    @classmethod
    def build(
        cls,
        rig: Rig,
        width: int,
        height: int,
        tile_size: int,
        *,
        band_rows: int = DEFAULT_BAND_ROWS,
    ) -> WarpPlan:
        """Evaluate the geometry once. Mirrors :func:`stitch_bands` step for step."""
        if width != 2 * height:
            raise ValueError(f"equirect output must be 2:1, got {width}x{height}")
        if tile_size < 1:
            raise ValueError(f"tile size must be positive, got {tile_size}")
        started = time.time()
        half = projection.half_extent(rig.fov_deg)
        views = rig.unique_views
        chunks: dict[int, list[tuple[I32, I32, F32]]] = {index: [] for index in views}

        for start in range(0, height, max(1, band_rows)):
            stop = min(start + band_rows, height)
            dirs = projection.equirect_directions(width, height, rows=slice(start, stop))
            base = start * width
            for index, view in views.items():
                x, y, visible = projection.project_to_tile(
                    dirs, view.yaw, view.elevation, rig.fov_deg, mirrored=rig.mirrored
                )
                if not visible.any():
                    continue
                column, row = projection.tile_pixel_index(
                    x[visible], y[visible], tile_size, rig.fov_deg
                )
                out_index = (np.flatnonzero(visible) + base).astype(np.int32)
                src_index = (row.astype(np.int32) * tile_size + column.astype(np.int32)).astype(
                    np.int32
                )
                chunks[index].append((out_index, src_index, _feather(x[visible], y[visible], half)))

        tiles = tuple(
            TileContributors(
                camera=index,
                out_index=np.concatenate([c[0] for c in parts]) if parts else np.zeros(0, np.int32),
                src_index=np.concatenate([c[1] for c in parts]) if parts else np.zeros(0, np.int32),
                weight=np.concatenate([c[2] for c in parts]) if parts else np.zeros(0, np.float32),
            )
            for index, parts in chunks.items()
        )
        return cls(
            width=width,
            height=height,
            tile_size=tile_size,
            fingerprint=plan_fingerprint(rig, width, height, tile_size),
            tiles=tiles,
            build_seconds=time.time() - started,
        )

    def _validate(self, tiles: dict[int, U8]) -> None:
        for tile in self.tiles:
            try:
                source = tiles[tile.camera]
            except KeyError:
                raise ValueError(f"missing tile for camera {tile.camera}") from None
            if source.shape != (self.tile_size, self.tile_size, 3):
                raise ValueError(
                    f"camera {tile.camera}: expected ({self.tile_size}, {self.tile_size}, 3), "
                    f"got {source.shape}"
                )

    def apply(
        self,
        tiles: dict[int, U8],
        *,
        with_stats: bool = False,
        threads: int = DEFAULT_THREADS,
    ) -> StitchResult:
        """Blend one frame's tiles. Same arithmetic as :func:`stitch_bands`.

        The output is split into `threads` horizontal bands processed concurrently. numpy
        releases the GIL for the gathers and scatters, so this is real parallelism: 8K
        goes from 3.7 s single-threaded to 1.7 s with four threads. Every output pixel
        still receives its tiles in the same order, so the result is identical for any
        thread count -- `tests/test_warp.py` asserts it.

        `with_stats` also accumulates the overlap-agreement sums for
        :func:`vr_compose.verify.agreement`; a sequence run samples it rather than paying
        for it on every frame.
        """
        self._validate(tiles)
        pixels = self.width * self.height
        colour = np.zeros((pixels, 3), np.float32)
        weight = np.zeros(pixels, np.float32)
        stats = BandStats.zeros(pixels if with_stats else 0)
        image = np.empty((pixels, 3), np.uint8)
        bands = max(1, min(int(threads), pixels))
        edges = np.linspace(0, pixels, bands + 1).astype(np.int64)
        flat = {tile.camera: tiles[tile.camera].reshape(-1, 3) for tile in self.tiles}

        def band(index: int) -> None:
            lo, hi = int(edges[index]), int(edges[index + 1])
            for tile in self.tiles:
                start = int(np.searchsorted(tile.out_index, lo))
                stop = int(np.searchsorted(tile.out_index, hi))
                if start == stop:
                    continue
                out_index = tile.out_index[start:stop]
                w = tile.weight[start:stop]
                sampled = flat[tile.camera][tile.src_index[start:stop]].astype(np.float32)
                colour[out_index] += sampled * w[:, None]
                weight[out_index] += w
                if with_stats:
                    luma = luma_bt709(sampled)
                    stats.count[out_index] += 1.0
                    stats.luma_sum[out_index] += luma
                    stats.luma_sq_sum[out_index] += luma.astype(np.float64) ** 2
            blended = colour[lo:hi] / np.maximum(weight[lo:hi], _EPSILON)[:, None]
            image[lo:hi] = np.rint(np.clip(blended, 0.0, 255.0)).astype(np.uint8)

        if bands == 1:
            band(0)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=bands) as pool:
                list(pool.map(band, range(bands)))
        return StitchResult(image=image.reshape(self.height, self.width, 3), stats=stats)

    def save(self, path: pathlib.Path) -> int:
        """Write the plan as an uncompressed .npz; returns bytes written."""
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, npt.NDArray[np.generic]] = {}
        for i, tile in enumerate(self.tiles):
            arrays[f"out_{i}"] = tile.out_index
            arrays[f"src_{i}"] = tile.src_index
            arrays[f"w_{i}"] = tile.weight
        meta = {
            "format": PLAN_FORMAT,
            "fingerprint": self.fingerprint,
            "width": self.width,
            "height": self.height,
            "tile_size": self.tile_size,
            "cameras": [t.camera for t in self.tiles],
            "build_seconds": self.build_seconds,
        }
        arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
        # numpy's stub types **kwds against `allow_pickle: bool`; the arrays are fine.
        np.savez(path, **arrays)  # type: ignore[arg-type]
        return path.stat().st_size

    @classmethod
    def load(cls, path: pathlib.Path, expected_fingerprint: str) -> WarpPlan:
        """Load a saved plan, refusing one built for different inputs."""
        with np.load(path) as archive:
            meta = json.loads(bytes(archive["meta"]).decode())
            if meta.get("format") != PLAN_FORMAT:
                raise ValueError(f"{path}: plan format {meta.get('format')} != {PLAN_FORMAT}")
            if meta["fingerprint"] != expected_fingerprint:
                raise ValueError(
                    f"{path}: plan was built for fingerprint {meta['fingerprint']}, "
                    f"need {expected_fingerprint}"
                )
            tiles = tuple(
                TileContributors(
                    camera=int(camera),
                    out_index=np.asarray(archive[f"out_{i}"], np.int32),
                    src_index=np.asarray(archive[f"src_{i}"], np.int32),
                    weight=np.asarray(archive[f"w_{i}"], np.float32),
                )
                for i, camera in enumerate(meta["cameras"])
            )
        return cls(
            width=int(meta["width"]),
            height=int(meta["height"]),
            tile_size=int(meta["tile_size"]),
            fingerprint=str(meta["fingerprint"]),
            tiles=tiles,
            build_seconds=float(meta["build_seconds"]),
        )
