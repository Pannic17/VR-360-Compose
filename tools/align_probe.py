"""Do the cameras of this render agree on where things are?

`rig_probe.py` asks which rig parameters fit best; this asks the blunter question the
parameters cannot answer: with the registered rig, do two cameras that see the same
direction draw it in the same place? Each camera of a pair is warped to the equirect on
its own and the overlap is phase-correlated patch by patch. A rig whose cameras share one
nodal point and sit at their nominal angles gives 0.00 px on essentially every patch.

    python tools/align_probe.py --source E:/22 --frame 1656            # the reference: 0.00
    python tools/align_probe.py --source E:/0910B --frame 500 --pairs all

What the answer means (AGENTS.md section 3):

* **0.00 px nearly everywhere** -- the rig is right and the blend has nothing to smear.
* **a shift with one direction, the same in every sector** -- a pose error. `rig_probe.py`
  can sometimes chase it down to a better FOV or ring elevation.
* **a shift that varies with position** -- parallax, or per-camera pose errors. No global
  parameter removes it; the blend will double every edge by that many pixels.

Read-only on the source directory.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import numpy.typing as npt

from vr_compose import io, projection, source
from vr_compose.rig import Rig, rig_for

F64 = npt.NDArray[np.float64]
DEFAULT_ROOT = pathlib.Path("E:/22")
PATCH = 128
"""Phase correlation window. Big enough to hold real structure, small enough that a
position-dependent shift does not average itself away."""
MIN_CONTRAST = 6.0
"""Below this a patch is flat water or flat sky and the correlation peak is noise."""


def warp_one(
    tiles: dict[int, npt.NDArray[np.uint8]], rig: Rig, index: int, size: int, width: int
) -> tuple[F64, npt.NDArray[np.bool_]]:
    """One camera's luma on the equirect grid, plus where it actually reaches."""
    height = width // 2
    dirs = projection.equirect_directions(width, height)
    view = rig.view_for(index)
    x, y, visible = projection.project_to_tile(
        dirs, view.yaw, view.elevation, rig.fov_deg, mirrored=rig.mirrored
    )
    px, py = projection.tile_pixel_position(x, y, size, rig.fov_deg)
    px = np.clip(np.nan_to_num(px, nan=0.0, posinf=0.0, neginf=0.0), 0, size - 2)
    py = np.clip(np.nan_to_num(py, nan=0.0, posinf=0.0, neginf=0.0), 0, size - 2)
    tile = tiles[index].astype(np.float64)
    luma = 0.2126 * tile[..., 0] + 0.7152 * tile[..., 1] + 0.0722 * tile[..., 2]
    ix, iy = np.floor(px).astype(np.int32), np.floor(py).astype(np.int32)
    fx, fy = px - ix, py - iy
    top = luma[iy, ix] * (1 - fx) + luma[iy, ix + 1] * fx
    bottom = luma[iy + 1, ix] * (1 - fx) + luma[iy + 1, ix + 1] * fx
    return (top * (1 - fy) + bottom * fy).reshape(height, width), visible.reshape(height, width)


def shifts(a: F64, b: F64, shared: npt.NDArray[np.bool_]) -> list[tuple[float, float, float]]:
    """`(magnitude, dy, dx)` per usable patch, in output pixels."""
    window = np.hanning(PATCH)[:, None] * np.hanning(PATCH)[None, :]
    height, width = a.shape
    found = []
    for y in range(0, height - PATCH, PATCH // 2):
        for x in range(0, width - PATCH, PATCH // 2):
            if not shared[y : y + PATCH, x : x + PATCH].all():
                continue
            left, right = a[y : y + PATCH, x : x + PATCH], b[y : y + PATCH, x : x + PATCH]
            if left.std() < MIN_CONTRAST:
                continue
            cross = np.fft.fft2((left - left.mean()) * window) * np.conj(
                np.fft.fft2((right - right.mean()) * window)
            )
            peak = np.fft.ifft2(cross / np.maximum(np.abs(cross), 1e-9)).real
            dy, dx = np.unravel_index(int(np.argmax(peak)), peak.shape)
            dy = dy - PATCH if dy > PATCH // 2 else dy
            dx = dx - PATCH if dx > PATCH // 2 else dx
            found.append((float(np.hypot(dx, dy)), float(dy), float(dx)))
    return found


def pairs_for(rig: Rig, choice: str) -> list[tuple[int, int]]:
    """`ring` (down against horizon, one per sector) or `all` (plus up against horizon)."""
    horizon = [i for i in rig.unique_indices if rig.view_for(i).elevation == 0.0]
    chosen = []
    for index in horizon:
        yaw = rig.view_for(index).yaw
        for elevation in (-45.0, 45.0) if choice == "all" else (-45.0,):
            partner = next(
                (
                    j
                    for j in rig.unique_indices
                    if rig.view_for(j).elevation == elevation and rig.view_for(j).yaw == yaw
                ),
                None,
            )
            if partner is not None:
                chosen.append((partner, index))
    return chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=pathlib.Path, default=DEFAULT_ROOT)
    parser.add_argument("--stem", default=None)
    parser.add_argument("--frame", type=int, default=None, help="default: the first frame")
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--pairs", choices=("ring", "all"), default="ring")
    args = parser.parse_args(argv)

    sets, searched = source.discover(args.source)
    if args.stem:
        sets = [s for s in sets if s.stem == args.stem]
    if len(sets) != 1:
        raise SystemExit(f"need exactly one source set (got {[s.stem for s in sets]}; {searched})")
    chosen = sets[0]
    rig = rig_for(chosen.camera_count)
    size = chosen.tile_size
    if size is None:
        raise SystemExit("source tiles are not square or not uniform")
    frame = args.frame if args.frame is not None else chosen.frames[0]
    degrees = 360.0 / args.width
    print(f"source : {chosen.root} stem {chosen.stem!r}, frame {frame}, tile {size}")
    print(f"scan   : {args.width} wide, 1 px = {degrees:.3f} deg, {PATCH} px patches\n")

    everything: list[float] = []
    for a, b in pairs_for(rig, args.pairs):
        tiles = io.load_tiles(chosen, frame, [a, b], workers=2)
        left, seen_left = warp_one(tiles, rig, a, size, args.width)
        right, seen_right = warp_one(tiles, rig, b, size, args.width)
        found = shifts(left, right, seen_left & seen_right)
        if not found:
            print(f"C{a:<2d} vs C{b:<2d}: no patch with enough contrast in the overlap")
            continue
        magnitude = np.array([f[0] for f in found])
        dy = np.array([f[1] for f in found])
        dx = np.array([f[2] for f in found])
        everything.extend(magnitude.tolist())
        print(
            f"C{a:<2d} vs C{b:<2d}: n {len(found):3d}  median {np.median(magnitude):5.2f} px  "
            f"max {magnitude.max():5.2f}  aligned {np.mean(magnitude < 0.5) * 100:5.1f}%  "
            f"dy {dy.mean():+5.2f}+-{dy.std():4.2f}  dx {dx.mean():+5.2f}+-{dx.std():4.2f}"
        )
    if everything:
        whole = np.array(everything)
        print(
            f"\noverall: median {np.median(whole):.2f} px ({np.median(whole) * degrees:.3f} deg), "
            f"{np.mean(whole < 0.5) * 100:.0f}% of patches aligned. "
            "A nodal rig at its nominal angles gives 0.00 px on 94-100%."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
