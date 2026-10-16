"""Which sampler *reconstructs the source* best? Round-trips the master back to a tile.

The agreement metric (metric A) cannot answer this. It measures how closely overlapping
tiles agree, so a blurrier reconstruction scores better simply by being smoother, and a
sharper kernel that overshoots at edges scores worse even where it is more faithful.
Measured: bilinear reaches a median of 0.484 and Catmull-Rom 0.524, which says nothing
about which one kept more of the render.

So this asks the question directly. The master is resampled *back* into a tile's own
pixel grid, along that tile's real ray directions, and compared with the tile the
renderer produced. Information the resample threw away cannot come back, so whichever
master reconstructs the tile best is the one that kept the most.

The return trip is bilinear in equirect space for every candidate, so it costs all of
them the same. It is only measured over the central half of each tile: further out the
master is a blend in which this tile is not dominant, and near the border the feather
weight has gone to zero, so a disagreement there is not the sampler's doing.

    python tools/fidelity_probe.py
    python tools/fidelity_probe.py --width 3840 --cameras 1 6 11

**The input sweep (P15 step 2).** With an EXR source the tiles are float levels, so they can
be the truth for their own 8-bit versions: `--inputs` stitches the same frame from the EXR
levels, from those levels quantised with an unbiased dither (the PNG the renderer would have
written), from a plain rounding, and from the real PNG capture, at 8 and 16 output bits, and
scores every master against the float tiles. `--bands` pools the error by the latitude each
tile pixel lands on, over all the tiles, for the central half and for the whole tile; the
whole-tile figures include the blend's own disagreement, equally for every input, so only
the differences between inputs are read there. That is the "pole headroom" question: how
much of what 8-bit quantisation threw away the master keeps when it is not thrown away.

    python tools/fidelity_probe.py --root <exr capture> --frame 571 --bands
        --inputs exr dither round png=<png capture>
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys
import time
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from vr_compose import io, source
from vr_compose.projection import camera_to_world, half_extent, tile_rays
from vr_compose.rig import Rig, rig_for
from vr_compose.stitch import SAMPLERS, Tile, stitch_frame
from vr_compose.warp import WarpPlan

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]
U8 = npt.NDArray[np.uint8]
Bool = npt.NDArray[np.bool_]
Panorama = npt.NDArray[np.uint8] | npt.NDArray[np.uint16]

DEFAULT_ROOT = pathlib.Path("E:/22")
CENTRAL = 0.5
"""Fraction of the tile's half-extent to score over. Outside it this tile is not the
dominant contributor to the master, so a difference is the blend's, not the sampler's."""
BANDS = (("cap >85", 85.0, 90.1), ("polar 70-85", 70.0, 85.0), ("mid 30-70", 30.0, 70.0),
         ("equator <30", 0.0, 30.0), ("all", 0.0, 90.1))  # fmt: skip
REGIONS = ("central 50%", "whole tile")


def sample_equirect(master: Panorama, lon: F64, lat: F64) -> F32:
    """Bilinear sample of an equirect image at the given directions.

    Longitude wraps; latitude clamps. The same for every candidate master, so it adds a
    constant blur to all of them rather than biasing the comparison.
    """
    height, width = master.shape[:2]
    u = ((lon / (2.0 * np.pi)) + 0.5) * width - 0.5
    v = (0.5 - lat / np.pi) * height - 0.5
    x0 = np.floor(u)
    y0 = np.clip(np.floor(v), 0, height - 2)
    fx = (u - x0).astype(np.float32)[:, None]
    fy = (v - y0).astype(np.float32)[:, None]
    c0 = x0.astype(np.int64) % width
    c1 = (x0.astype(np.int64) + 1) % width
    r0 = y0.astype(np.int64)
    top = master[r0, c0].astype(np.float32) * (1.0 - fx) + master[r0, c1].astype(np.float32) * fx
    bottom = (
        master[r0 + 1, c0].astype(np.float32) * (1.0 - fx)
        + master[r0 + 1, c1].astype(np.float32) * fx
    )
    return np.asarray(top * (1.0 - fy) + bottom * fy, dtype=np.float32)


def psnr(a: F32, b: F32) -> float:
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0.0 else 10.0 * np.log10(255.0**2 / mse)


# -- the input sweep -----------------------------------------------------------------------


def dithered(tiles: Mapping[int, Tile], seed: int) -> dict[int, U8]:
    """Unbiased uniform dither, then floor: the same picture at 8 bits, no systematic offset."""
    rng = np.random.default_rng(seed)
    out: dict[int, U8] = {}
    for camera, tile in tiles.items():
        noise = rng.random(tile.shape, dtype=np.float32)
        out[camera] = np.asarray(
            np.clip(np.floor(tile.astype(np.float32) + noise), 0, 255), dtype=np.uint8
        )
    return out


def rounded(tiles: Mapping[int, Tile]) -> dict[int, U8]:
    return {camera: np.asarray(np.rint(tile), np.uint8) for camera, tile in tiles.items()}


def png_capture(path: pathlib.Path, frame: int, cameras: list[int]) -> dict[int, U8]:
    found = [s for s in source.scan(path) if s.usable and s.tile_format == "png"]
    if len(found) != 1:
        raise SystemExit(f"{path}: expected one usable PNG source set, found {len(found)}")
    if frame not in found[0].frames:
        raise SystemExit(f"{path}: frame {frame} is not in the PNG capture")
    return io.uint8_tiles(io.load_tiles(found[0], frame, cameras, workers=8))


@dataclasses.dataclass(slots=True)
class TileGeometry:
    """Where each pixel of one tile lands: its direction, latitude band and region masks."""

    lon: F64
    lat: F64
    band: npt.NDArray[np.int64]
    """Index into `BANDS` (without the trailing "all") per pixel."""
    central: Bool


def tile_geometry(rig: Rig, size: int, camera: int) -> TileGeometry:
    rays = tile_rays(size, rig.fov_deg, mirrored=rig.mirrored)
    view = rig.view_for(camera)
    world = camera_to_world(view.yaw, view.elevation) @ rays
    lon = np.arctan2(world[1], world[0])
    lat = np.arcsin(np.clip(world[2], -1.0, 1.0))
    degrees = np.abs(np.degrees(lat))
    band = np.full(lat.shape, -1, np.int64)
    for k, (_, lo, hi) in enumerate(BANDS[:-1]):
        band[(degrees >= lo) & (degrees < hi)] = k
    grid = (np.arange(size, dtype=np.float64) + 0.5) / size * 2.0 - 1.0
    gx, gy = np.meshgrid(grid, grid)
    central = ((np.abs(gx) < CENTRAL) & (np.abs(gy) < CENTRAL)).reshape(-1)
    return TileGeometry(lon, lat, band, central)


class Pool:
    """Squared and absolute error pooled per (region, band) over every tile scored."""

    def __init__(self) -> None:
        self.sq = np.zeros((len(REGIONS), len(BANDS)), np.float64)
        self.ab = np.zeros((len(REGIONS), len(BANDS)), np.float64)
        self.n = np.zeros((len(REGIONS), len(BANDS)), np.float64)

    def add(self, error: F32, geometry: TileGeometry) -> None:
        e = error.astype(np.float64)
        sq = (e * e).sum(axis=1)
        ab = np.abs(e).sum(axis=1)
        whole = np.ones_like(geometry.central)
        for r, region in enumerate((geometry.central, whole)):
            for k in range(len(BANDS) - 1):
                mask = region & (geometry.band == k)
                self.sq[r, k] += sq[mask].sum()
                self.ab[r, k] += ab[mask].sum()
                self.n[r, k] += 3.0 * float(mask.sum())
            self.sq[r, -1] += sq[region].sum()
            self.ab[r, -1] += ab[region].sum()
            self.n[r, -1] += 3.0 * float(region.sum())

    def psnr(self, region: int, band: int) -> float:
        if self.n[region, band] == 0:
            return float("nan")
        mse = self.sq[region, band] / self.n[region, band]
        return float("inf") if mse == 0.0 else float(10.0 * np.log10(255.0**2 / mse))

    def mean_abs(self, region: int, band: int) -> float:
        if self.n[region, band] == 0:
            return float("nan")
        return float(self.ab[region, band] / self.n[region, band])


def score_master(
    master: Panorama, truth: Mapping[int, Tile], geometries: dict[int, TileGeometry]
) -> Pool:
    scale = np.float32(257.0) if master.dtype == np.uint16 else np.float32(1.0)
    pool = Pool()
    for camera, geometry in geometries.items():
        reconstructed = sample_equirect(master, geometry.lon, geometry.lat) / scale
        original = truth[camera].reshape(-1, 3).astype(np.float32)
        pool.add(reconstructed - original, geometry)
    return pool


def score_tiles(
    tiles: Mapping[int, Tile], truth: Mapping[int, Tile], geometries: dict[int, TileGeometry]
) -> Pool:
    """The quantisation alone: an input's tiles against the truth, no resample."""
    pool = Pool()
    for camera, geometry in geometries.items():
        pool.add(
            tiles[camera].reshape(-1, 3).astype(np.float32)
            - truth[camera].reshape(-1, 3).astype(np.float32),
            geometry,
        )
    return pool


def print_table(title: str, rows: list[tuple[str, Pool]], region: int, value: str) -> None:
    print(f"\n  {title} -- {REGIONS[region]}")
    print("  " + f"{'input':<12}" + "".join(f"{name:>14}" for name, _, _ in BANDS))
    for label, pool in rows:
        cells = []
        for k in range(len(BANDS)):
            number = pool.psnr(region, k) if value == "psnr" else pool.mean_abs(region, k)
            cells.append(f"{number:14.3f}" if np.isfinite(number) else f"{'-':>14}")
        print("  " + f"{label:<12}" + "".join(cells))


def input_sweep(
    args: argparse.Namespace,
    chosen: source.SourceSet,
    rig: Rig,
    size: int,
    truth: Mapping[int, Tile],
) -> int:
    if chosen.tile_format != "exr":
        print("--inputs needs an EXR source: its float levels are the truth", file=sys.stderr)
        return 2
    cameras = list(rig.unique_indices) if args.bands else default_cameras(rig)
    inputs: list[tuple[str, Mapping[int, Tile]]] = []
    for name in args.inputs:
        if name == "exr":
            inputs.append(("exr", truth))
        elif name == "dither":
            inputs.append(("dither", dithered(truth, seed=args.frame)))
        elif name == "round":
            inputs.append(("round", rounded(truth)))
        elif name.startswith("png="):
            inputs.append(("png", png_capture(pathlib.Path(name[4:]), args.frame, cameras)))
        else:
            raise SystemExit(f"unknown input {name!r}; exr, dither, round or png=<capture>")
    started = time.time()
    plan = WarpPlan.build(rig, args.width, args.width // 2, size)
    geometries = {camera: tile_geometry(rig, size, camera) for camera in cameras}
    print(
        f"input sweep: truth = EXR float levels; master {args.width}x{args.width // 2} "
        f"{plan.sampler}, feather {plan.feather_power:g}, seam band {plan.seam_band:g}, no "
        f"harmoniser; tiles {cameras}; plan {time.time() - started:.1f} s"
    )
    rows: list[tuple[str, Pool]] = []
    for name, tiles in inputs:
        for bit_depth in args.bit_depths:
            tick = time.time()
            master = plan.apply(tiles, threads=8, bit_depth=bit_depth).image
            pool = score_master(master, truth, geometries)
            rows.append((f"{name} {bit_depth}b", pool))
            print(f"  {name} {bit_depth}-bit: {time.time() - tick:.1f} s")
    floors = [
        (f"{name} tile", score_tiles(tiles, truth, geometries))
        for name, tiles in inputs
        if name != "exr"
    ]
    for region in range(len(REGIONS)):
        print_table("PSNR dB of the round trip against the float tiles", rows, region, "psnr")
    print_table("mean |error| levels of the round trip", rows, 1, "abs")
    print_table("quantisation alone: input tile vs float tile, PSNR dB", floors, 1, "psnr")
    print(
        "\n  Read: the same geometry and blend for every row, so a row's gain over 'dither' is\n"
        "  what the float source keeps; whole-tile columns include the blend's disagreement\n"
        "  equally for every row."
    )
    return 0


def default_cameras(rig: Rig) -> list[int]:
    """One tile per elevation ring: the regimes differ by latitude."""
    by_elevation: dict[float, int] = {}
    for index in rig.unique_indices:
        by_elevation.setdefault(rig.view_for(index).elevation, index)
    return sorted(by_elevation.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=DEFAULT_ROOT)
    parser.add_argument("--frame", type=int, default=None, help="default: 1656, else the first")
    parser.add_argument("--width", type=int, default=7680, help="master width; native is fairest")
    parser.add_argument(
        "--feathers",
        type=float,
        nargs="+",
        default=None,
        help="sweep feather exponents at the default sampler instead of sweeping samplers",
    )
    parser.add_argument(
        "--cameras",
        type=int,
        nargs="+",
        default=None,
        help="which tiles to score against (default: one per elevation ring)",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help="P15: sweep the source instead of the sampler: exr, dither, round, png=<capture>",
    )
    parser.add_argument("--bit-depths", type=int, nargs="+", default=[16, 8])
    parser.add_argument("--bands", action="store_true", help="pool by latitude, all tiles")
    args = parser.parse_args(argv)

    found = [s for s in source.scan(args.root) if s.usable]
    if not found:
        print(f"no usable source set in {args.root}", file=sys.stderr)
        return 2
    chosen = found[0]
    rig = rig_for(chosen.camera_count)
    size = chosen.tile_size
    if size is None:
        print("source tiles are not square or not uniform", file=sys.stderr)
        return 2
    if args.frame is None:
        args.frame = 1656 if 1656 in chosen.frames else chosen.frames[0]
    tiles = io.load_tiles(chosen, args.frame, list(rig.unique_indices), workers=8)
    if args.inputs:
        return input_sweep(args, chosen, rig, size, tiles)

    # one tile per elevation ring by default: the regimes differ by latitude
    if args.cameras:
        cameras = [c for c in args.cameras if c in rig.unique_indices]
    else:
        cameras = default_cameras(rig)

    half = half_extent(rig.fov_deg)
    rays = tile_rays(size, rig.fov_deg, mirrored=rig.mirrored)
    grid = (np.arange(size, dtype=np.float64) + 0.5) / size * 2.0 - 1.0
    gx, gy = np.meshgrid(grid, grid)
    central = ((np.abs(gx) < CENTRAL) & (np.abs(gy) < CENTRAL)).reshape(-1)

    print(
        f"frame {args.frame}, master {args.width}x{args.width // 2}, "
        f"tiles {cameras}, scored over the central {CENTRAL:.0%} of each tile\n"
    )
    label = "feather" if args.feathers else "sampler"
    print(f"  {label:<11} {'stitch':>8}  " + "  ".join(f"cam{c:<7}" for c in cameras) + "  mean")
    variants: list[tuple[str, dict[str, object]]] = (
        [(f"{power:g}", {"feather_power": power}) for power in args.feathers]
        if args.feathers
        else [(sampler, {"sampler": sampler}) for sampler in SAMPLERS]
    )
    for name, options in variants:
        started = time.time()
        master = stitch_frame(tiles, rig, args.width, **options).image  # type: ignore[arg-type]
        elapsed = time.time() - started
        scores = []
        for camera in cameras:
            view = rig.view_for(camera)
            world = camera_to_world(view.yaw, view.elevation) @ rays
            lon = np.arctan2(world[1], world[0])
            lat = np.arcsin(np.clip(world[2], -1.0, 1.0))
            reconstructed = sample_equirect(master, lon[central], lat[central])
            original = tiles[camera].reshape(-1, 3)[central].astype(np.float32)
            scores.append(psnr(reconstructed, original))
        print(
            f"  {name:<11} {elapsed:7.1f}s  "
            + "  ".join(f"{s:7.2f} dB" for s in scores)
            + f"  {float(np.mean(scores)):7.2f} dB"
        )
    print(
        "\nHigher is better: it is the master reconstructing the render it came from.\n"
        f"(half extent {half:g}; the return trip is bilinear for every row, so it is a\n"
        "constant cost, not a bias.)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
