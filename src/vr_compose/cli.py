"""Command-line entry point.

The GUI in P5 must be a thin shell over exactly this code, so every operation lives in a
library module and this file only parses arguments and prints. Nothing here hardcodes a
source path: `--source` is optional and discovery fills it in (AGENTS.md §2, constraint 3).
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from collections.abc import Sequence

from vr_compose import __version__, io, verify
from vr_compose import source as source_mod
from vr_compose.rig import UnknownRigError, rig_for
from vr_compose.stitch import DEFAULT_BAND_ROWS, stitch_frame

DEFAULT_WIDTH = 7680


def _resolve_source(explicit: pathlib.Path | None) -> source_mod.SourceSet:
    """Pick exactly one source set, or exit with something actionable."""
    found, searched = source_mod.discover(explicit)
    if not found:
        print("no usable source set found. searched:", file=sys.stderr)
        for path in searched[:12]:
            print(f"  {path}", file=sys.stderr)
        if len(searched) > 12:
            print(f"  ... and {len(searched) - 12} more", file=sys.stderr)
        if explicit is not None:
            for candidate in source_mod.scan(explicit):
                print(f"\nrejected {explicit} / stem {candidate.stem!r}:", file=sys.stderr)
                for problem in candidate.problems:
                    print(f"  - {problem}", file=sys.stderr)
        print(
            "\nA source directory needs at least two `CameraN` directories whose PNGs are\n"
            "named `<stem>.<frame>.png`, either directly inside or one level below.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if len(found) > 1:
        stems = ", ".join(repr(s.stem) for s in found)
        print(
            f"{len(found)} source sets in {found[0].root} (stems: {stems}).\n"
            "Choose one with --stem; processing an arbitrary first match would risk the "
            "wrong eye.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return found[0]


def _select_stem(candidates: list[source_mod.SourceSet], stem: str | None) -> source_mod.SourceSet:
    if stem is None:
        return candidates[0]
    for candidate in candidates:
        if candidate.stem == stem:
            return candidate
    available = ", ".join(repr(c.stem) for c in candidates)
    raise SystemExit(f"no stem {stem!r} here; available: {available}")


def cmd_discover(args: argparse.Namespace) -> int:
    found, searched = source_mod.discover(args.source)
    print(f"searched {len(searched)} location(s)\n")
    if not found:
        for path in searched[:12]:
            print(f"  {path}")
        rejected = source_mod.scan(args.source) if args.source else []
        for candidate in rejected:
            print(f"\nrejected stem {candidate.stem!r}:")
            for problem in candidate.problems:
                print(f"  - {problem}")
        return 1
    for candidate in found:
        print(candidate.describe())
        try:
            rig = rig_for(candidate.camera_count)
            print(
                f"rig        : {rig.name}  fov={rig.fov_deg:g}deg  "
                f"{len(rig.unique_indices)} distinct of {rig.file_count} files"
            )
            if rig.duplicate_indices:
                print(f"redundant  : cameras {list(rig.duplicate_indices)} duplicate other views")
        except UnknownRigError as exc:
            print(f"rig        : UNKNOWN -- {exc}")
        print()
    return 0


def cmd_frame(args: argparse.Namespace) -> int:
    candidates = [s for s in source_mod.scan(args.source) if s.usable] if args.source else []
    chosen = _select_stem(candidates, args.stem) if candidates else _resolve_source(args.source)

    try:
        rig = rig_for(chosen.camera_count)
    except UnknownRigError as exc:
        raise SystemExit(str(exc)) from None

    frames = chosen.frames
    frame = args.frame if args.frame is not None else frames[0]
    if frame not in frames:
        near = ", ".join(str(f) for f in frames[:5])
        raise SystemExit(
            f"frame {frame} is not present in every camera. "
            f"{len(frames)} frames available, starting {near}..."
        )

    indices = list(rig.unique_indices)
    print(f"source     : {chosen.root}  stem {chosen.stem!r}")
    print(f"rig        : {rig.name}, reading {len(indices)} of {rig.file_count} files")
    started = time.time()
    tiles = io.load_tiles(chosen, frame, indices, workers=args.decode_workers)
    decoded = time.time() - started

    result = stitch_frame(tiles, rig, args.width, band_rows=args.band_rows)
    elapsed = time.time() - started
    height = args.width // 2
    print(
        f"stitched   : frame {frame} at {args.width}x{height} in {elapsed:.1f} s "
        f"(decode {decoded:.1f} s), overlap {result.overlap_fraction:.1%}, "
        f"up to {result.max_contributors} tiles"
    )
    print(f"wrap seam  : {verify.wrap_seam_error(result.image):.2f} / 255")
    report = verify.agreement(result.stats)
    print(report.report())

    if args.out is not None:
        size = io.write_png(args.out, result.image, compress_level=args.compress_level)
        print(f"wrote      : {args.out}  ({size / 2**20:.1f} MiB)")
    return 0 if report.passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vr-compose", description="VR 360 composition toolkit")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--source",
        type=pathlib.Path,
        default=None,
        help="source directory; omitted means search next to the executable",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    discover = sub.add_parser("discover", help="report the detected source sets and rig")
    discover.set_defaults(func=cmd_discover)

    frame = sub.add_parser("frame", help="stitch a single frame and check its geometry")
    frame.add_argument("--frame", type=int, default=None, help="default: the first shared frame")
    frame.add_argument("--stem", default=None, help="which eye/scene, when a directory has several")
    frame.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="output width (2:1)")
    frame.add_argument("--out", type=pathlib.Path, default=None, help="write the panorama here")
    frame.add_argument("--compress-level", type=int, default=6, choices=range(10))
    frame.add_argument("--band-rows", type=int, default=DEFAULT_BAND_ROWS)
    frame.add_argument("--decode-workers", type=int, default=8)
    frame.set_defaults(func=cmd_frame)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result
