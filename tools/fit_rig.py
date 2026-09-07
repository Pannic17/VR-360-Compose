"""Solve a camera rig by matching each tile against a finished equirect panorama.

This is how the registered rig was derived (AGENTS.md §3), and it is what to run when a
source set has a camera count `vr_compose.rig.REGISTRY` does not know. It needs no
calibration target and no feature matching: a finished panorama of the same scene *is*
the ground truth, so a coarse-to-fine NCC search over (yaw, elevation, roll, FOV)
recovers each orientation.

For the reference data there is no frame that both the surviving tiles and the finished
output share, so adjacent frames are used (1655 output against 1656 tiles). The one frame
of scene motion costs a little accuracy but not the solution.

    python tools/fit_rig.py --cameras 1 2 3
    python tools/fit_rig.py --source E:/22 --equirect-frame 1655 --tile-frame 1656
"""

from __future__ import annotations

import argparse
import itertools
import pathlib
import sys

import numpy as np
import numpy.typing as npt
from PIL import Image

from vr_compose.projection import camera_to_world, tile_rays
from vr_compose.rig import UnknownRigError, rig_for
from vr_compose.source import scan

Image.MAX_IMAGE_PIXELS = None

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]

DEFAULT_ROOT = pathlib.Path("E:/22")


class Matcher:
    """Scores a candidate orientation by NCC against a downsampled equirect."""

    def __init__(self, equirect: pathlib.Path, width: int) -> None:
        height = width // 2
        with Image.open(equirect) as image:
            resized = image.convert("L").resize((width, height), Image.Resampling.BILINEAR)
        self.panorama: F32 = np.asarray(resized, np.float32)
        self.width = width
        self.height = height

    def sample(self, dirs: F64) -> F32:
        lon = np.arctan2(dirs[1], dirs[0])
        lat = np.arcsin(np.clip(dirs[2], -1.0, 1.0))
        u = np.clip((lon / (2 * np.pi) + 0.5) * self.width, 0, self.width - 1).astype(np.int32)
        v = np.clip((0.5 - lat / np.pi) * self.height, 0, self.height - 1).astype(np.int32)
        return np.asarray(self.panorama[v, u], np.float32)

    def score(self, tile: F32, rays: F64, yaw: float, elevation: float, roll: float) -> float:
        rotation = camera_to_world(yaw, elevation) @ _roll_matrix(roll)
        return _ncc(tile, self.sample(rotation @ rays))


def _roll_matrix(deg: float) -> F64:
    """Roll about the camera's own forward (+X) axis."""
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ncc(a: F32, b: F32) -> float:
    """Zero-mean normalised cross-correlation: robust to exposure differences."""
    am, bm = a - a.mean(), b - b.mean()
    na, nb = float(np.linalg.norm(am)), float(np.linalg.norm(bm))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(am @ bm / (na * nb))


def load_tile(path: pathlib.Path, size: int) -> F32:
    with Image.open(path) as image:
        resized = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(resized, np.float32).reshape(-1)


def search(
    matcher: Matcher,
    tile_file: pathlib.Path,
    size: int,
    fovs: list[float],
    yaws: list[float],
    elevations: list[float],
    rolls: list[float],
    mirrored: bool,
) -> tuple[float, dict[str, float]]:
    tile = load_tile(tile_file, size)
    best_score = -2.0
    best: dict[str, float] = {}
    for fov in fovs:
        rays = tile_rays(size, fov, mirrored=mirrored)
        for yaw, elevation, roll in itertools.product(yaws, elevations, rolls):
            score = matcher.score(tile, rays, yaw, elevation, roll)
            if score > best_score:
                best_score = score
                best = {"fov": fov, "yaw": yaw, "elevation": elevation, "roll": roll}
    return best_score, best


def _around(centre: float, step: float, count: int = 2) -> list[float]:
    return [centre + k * step for k in range(-count, count + 1)]


def solve(
    matcher: Matcher, tile_file: pathlib.Path, mirrored: bool
) -> tuple[float, dict[str, float]]:
    """Coarse global sweep, then two refinement passes."""
    _, coarse = search(
        matcher,
        tile_file,
        48,
        [float(f) for f in range(60, 145, 5)],
        [float(y) for y in range(0, 360, 6)],
        [float(e) for e in range(-84, 85, 6)],
        [0.0],
        mirrored,
    )
    _, mid = search(
        matcher,
        tile_file,
        96,
        _around(coarse["fov"], 2.0),
        _around(coarse["yaw"], 2.0),
        _around(coarse["elevation"], 2.0),
        [float(r) for r in range(-20, 21, 5)],
        mirrored,
    )
    return search(
        matcher,
        tile_file,
        96,
        _around(mid["fov"], 0.5),
        _around(mid["yaw"], 0.5),
        _around(mid["elevation"], 0.5),
        _around(mid["roll"], 1.0),
        mirrored,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=pathlib.Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--equirect",
        type=pathlib.Path,
        default=None,
        help="reference panorama; default: --equirect-frame in the output dir",
    )
    parser.add_argument("--equirect-frame", type=int, default=1655)
    parser.add_argument("--tile-frame", type=int, default=1656)
    parser.add_argument(
        "--cameras",
        type=int,
        nargs="*",
        default=None,
        help="default: one per distinct viewpoint of the registered rig",
    )
    parser.add_argument("--equirect-width", type=int, default=2048)
    args = parser.parse_args(argv)

    usable = [candidate for candidate in scan(args.source) if candidate.usable]
    if not usable:
        raise SystemExit(f"no usable source set in {args.source}")
    detected = usable[0]

    equirect = args.equirect
    if equirect is None:
        if detected.output_dir is None:
            raise SystemExit("no output directory found; pass --equirect explicitly")
        matches = sorted(detected.output_dir.glob(f"*.{args.equirect_frame:04d}.png"))
        if not matches:
            raise SystemExit(
                f"no panorama for frame {args.equirect_frame} in {detected.output_dir}"
            )
        equirect = matches[0]

    expected = None
    try:
        expected = rig_for(detected.camera_count)
    except UnknownRigError:
        print(f"no registered rig for {detected.camera_count} cameras -- solving from scratch\n")

    cameras = args.cameras
    if cameras is None:
        cameras = (
            list(expected.unique_indices) if expected else list(range(1, detected.camera_count + 1))
        )

    matcher = Matcher(equirect, args.equirect_width)
    print(f"reference {equirect.name} vs tiles {args.tile_frame}\n")

    probe = detected.tile_path(cameras[0], args.tile_frame)
    print(f"resolving handedness on Camera{cameras[0]} ...")
    handed = {
        mirrored: search(
            matcher,
            probe,
            48,
            [float(f) for f in range(60, 145, 10)],
            [float(y) for y in range(0, 360, 6)],
            [float(e) for e in range(-84, 85, 6)],
            [0.0],
            mirrored,
        )[0]
        for mirrored in (False, True)
    }
    mirrored = max(handed, key=lambda key: handed[key])
    print(
        f"  mirrored=False ncc={handed[False]:.4f}   mirrored=True ncc={handed[True]:.4f}"
        f"   -> mirrored={mirrored}\n"
    )

    header = f"{'camera':<9} {'ncc':>7} {'fov':>7} {'yaw':>8} {'elev':>7} {'roll':>6}"
    print(header + ("   expected (registered rig)" if expected else ""))
    deviations: list[float] = []
    for camera in cameras:
        tile_file = detected.tile_path(camera, args.tile_frame)
        if not tile_file.exists():
            print(f"Camera{camera:<3d} SKIPPED (no frame {args.tile_frame})")
            continue
        score, fit = solve(matcher, tile_file, mirrored)
        line = (
            f"Camera{camera:<3d} {score:7.4f} {fit['fov']:7.1f} {fit['yaw']:8.1f} "
            f"{fit['elevation']:7.1f} {fit['roll']:6.1f}"
        )
        if expected is not None:
            want = expected.view_for(camera)
            yaw_error = (fit["yaw"] - want.yaw + 180.0) % 360.0 - 180.0
            elevation_error = fit["elevation"] - want.elevation
            deviations += [abs(yaw_error), abs(elevation_error)]
            line += (
                f"   yaw={want.yaw:5.1f} elev={want.elevation:+5.1f}"
                f"  (dyaw={yaw_error:+5.1f} delev={elevation_error:+5.1f})"
            )
        print(line)

    if deviations:
        print(
            f"\nmax deviation from the registered rig: {max(deviations):.1f} deg. "
            "The search grid is 0.5 deg, so anything under about 1 deg confirms it."
        )
    else:
        print("\nAdd the solved orientations to vr_compose.rig.REGISTRY to use this layout.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
