"""Do the cameras of this render agree on where things are?

`rig_probe.py` asks which rig parameters fit best; this asks the blunter question the
parameters cannot answer: with the registered rig, do two cameras that see the same
direction draw it in the same place? Each camera of a pair is warped to the equirect on
its own and the overlap is phase-correlated patch by patch. A rig whose cameras share one
nodal point and sit at their nominal angles gives 0.00 px on essentially every patch.

    python tools/align_probe.py --source E:/360/0904 --frame 1656      # the reference: 0.00
    python tools/align_probe.py --source E:/360/0910B --frame 500 --pairs all

The measurement itself lives in :mod:`vr_compose.layout` since P11b, where it is the
layout gate a 15-camera run passes before stitching; this is the same code with every
pair printed and no verdict. `--pairs all` is exactly the gate's ten pairs.

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

from vr_compose import io, layout, projection, source
from vr_compose.rig import Rig, rig_for

F64 = npt.NDArray[np.float64]
DEFAULT_ROOT = pathlib.Path("E:/360/0904")


def warp_one(
    tiles: dict[int, npt.NDArray[np.uint8]], rig: Rig, index: int, size: int, width: int
) -> tuple[F64, npt.NDArray[np.bool_]]:
    """One camera's luma on the whole equirect grid, plus where it actually reaches.

    The full-frame form `blend_probe.py` needs; the gate warps only each pair's overlap.
    """
    del size  # the tile carries it; kept so the signature callers know stays put
    height = width // 2
    luma, visible = layout.warp_luma(
        tiles[index], rig, index, projection.equirect_directions(width, height)
    )
    return luma.reshape(height, width), visible.reshape(height, width)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=pathlib.Path, default=DEFAULT_ROOT)
    parser.add_argument("--stem", default=None)
    parser.add_argument("--frame", type=int, default=None, help="default: the first frame")
    parser.add_argument(
        "--width", type=int, default=None, help="default: the layout gate's analysis width"
    )
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
    width = args.width or layout.analysis_width(rig, size)
    degrees = 360.0 / width
    print(f"source : {chosen.root} stem {chosen.stem!r}, frame {frame}, tile {size}")
    print(f"scan   : {width} wide, 1 px = {degrees:.3f} deg, {layout.PATCH} px patches\n")

    tiles = io.load_tiles(chosen, frame, list(rig.unique_indices), workers=8)
    everything: list[float] = []
    for a, b in layout.pairs_for(rig, args.pairs):
        pair = layout.pair_shifts(tiles, rig, a, b, width)
        print(pair.line())
        everything.extend(s[0] for s in pair.shifts)
    if everything:
        whole = np.array(everything)
        print(
            f"\noverall: median {np.median(whole):.2f} px ({np.median(whole) * degrees:.3f} deg), "
            f"{np.mean(whole < layout.ALIGNED_PX) * 100:.0f}% of patches aligned. "
            "A nodal rig at its nominal angles gives 0.00 px on 94-100%."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
