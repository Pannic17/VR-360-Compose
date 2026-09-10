"""Which rig do these tiles actually fit? Scan FOV and ring elevations by overlap agreement.

The geometry gate (metric A) failed on a scene rendered elsewhere with median 1.19, mean
2.91, and **10 % of overlap directions disagreeing by more than 8 levels**. Measured on
the reference data (2026-09-09), that last number separates the two possible causes:

- view-dependent *content* (exposure, fog, bloom) moves the median and mean but leaves the
  over-8 share near 0.04-0.08 %;
- a *geometric* mismatch of the rig -- the tiles rendered at 90.5-91 degrees while the rig
  assumes 90 -- gives median 1.5-2.0, mean 2.7-3.6 and an over-8 share of 6-10 %.

So this scans the rig parameters and reports metric A for each candidate: the true value
is where the tiles agree best, no reference panorama needed. It is the cheap complement to
`fit_rig.py`, which solves every camera independently against a finished equirect.

    python tools/rig_probe.py --source D:/CR/0909C --frame 100
    python tools/rig_probe.py --source E:/22 --frame 1656 --fov 88 92 0.5     # must find 90

Read-only on the source directory.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import time

from vr_compose import io, source, verify
from vr_compose.rig import Rig, View, rig_for
from vr_compose.stitch import stitch_frame

DEFAULT_ROOT = pathlib.Path("E:/22")


def frange(start: float, stop: float, step: float) -> list[float]:
    values = []
    x = start
    while x <= stop + 1e-9:
        values.append(round(x, 4))
        x += step
    return values


def score(rig: Rig, tiles: dict, width: int, sampler: str) -> verify.Agreement:  # type: ignore[type-arg]
    return verify.agreement(stitch_frame(tiles, rig, width, sampler=sampler).stats)


def row(label: str, report: verify.Agreement, seconds: float) -> str:
    return (
        f"{label:<30} median {report.median:5.2f}  mean {report.mean:5.2f}  "
        f"p95 {report.p95:6.2f}  >8 {report.fraction_over_8:7.3%}  ({seconds:4.1f}s)"
    )


def with_ring_offset(rig: Rig, delta: float) -> Rig:
    """Move the upper ring up and the lower ring down by `delta` degrees."""
    views = tuple(
        View(
            v.yaw, v.elevation + (delta if v.elevation > 0 else -delta if v.elevation < 0 else 0.0)
        )
        for v in rig.views
    )
    return dataclasses.replace(rig, views=views)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=pathlib.Path, default=DEFAULT_ROOT)
    parser.add_argument("--stem", default=None)
    parser.add_argument("--frame", type=int, default=None, help="default: the first frame")
    parser.add_argument("--width", type=int, default=3840, help="stitch width for the scan")
    parser.add_argument("--sampler", default="bilinear")
    parser.add_argument(
        "--fov", type=float, nargs=3, default=(88.0, 92.0, 0.5), metavar=("LO", "HI", "STEP")
    )
    parser.add_argument(
        "--ring", type=float, nargs=3, default=(-2.0, 2.0, 1.0), metavar=("LO", "HI", "STEP"),
        help="offset of the +-45 degree rings, scanned at the best FOV",
    )  # fmt: skip
    args = parser.parse_args(argv)

    sets, searched = source.discover(args.source)
    if args.stem:
        sets = [s for s in sets if s.stem == args.stem]
    if len(sets) != 1:
        raise SystemExit(
            f"need exactly one source set (got {[s.stem for s in sets]}; searched {searched})"
        )
    chosen = sets[0]
    rig = rig_for(chosen.camera_count)
    frame = args.frame if args.frame is not None else chosen.frames[0]
    tiles = io.load_tiles(chosen, frame, list(rig.unique_indices), workers=8)
    print(f"source : {chosen.root} stem {chosen.stem!r}, frame {frame}, tile {chosen.tile_size}")
    print(f"rig    : {rig.name}, registered fov {rig.fov_deg}, mirrored {rig.mirrored}")
    print(f"scan   : {args.width}x{args.width // 2}, sampler {args.sampler}\n")

    print("== FOV ==")
    best_fov, best_median = rig.fov_deg, float("inf")
    for fov in frange(*args.fov):
        started = time.perf_counter()
        report = score(dataclasses.replace(rig, fov_deg=fov), tiles, args.width, args.sampler)
        mark = ""
        if report.median < best_median:
            best_fov, best_median, mark = fov, report.median, "  <-- best so far"
        print(row(f"fov {fov:.2f}", report, time.perf_counter() - started) + mark)

    print(f"\n== ring elevation offset at fov {best_fov:.2f} ==")
    base = dataclasses.replace(rig, fov_deg=best_fov)
    for delta in frange(*args.ring):
        started = time.perf_counter()
        report = score(with_ring_offset(base, delta), tiles, args.width, args.sampler)
        print(row(f"rings {delta:+.1f} deg", report, time.perf_counter() - started))

    print("\n== mirroring ==")
    for mirrored in (True, False):
        started = time.perf_counter()
        report = score(
            dataclasses.replace(base, mirrored=mirrored), tiles, args.width, args.sampler
        )
        print(row(f"mirrored={mirrored}", report, time.perf_counter() - started))

    print(
        f"\nbest fov {best_fov:.2f} (median {best_median:.2f}); registered {rig.fov_deg}. "
        "If they differ, the tiles were rendered with a different camera FOV than the rig assumes."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
