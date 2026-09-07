"""Survey a source set and reproduce the numbers in AGENTS.md §4 and §8.

What `vr-compose discover` does not cover: the alpha-channel finding, pixel-level
confirmation that the rig's duplicate groups really are duplicates, and the previous
pipeline's throughput derived from output file timestamps. Reads only.

    python tools/inspect_data.py --skip-duplicates      # header-only, seconds
    python tools/inspect_data.py --source E:/22
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sys

import numpy as np
import numpy.typing as npt
from PIL import Image

from vr_compose.rig import UnknownRigError, rig_for
from vr_compose.source import SourceSet, png_dimensions, scan

Image.MAX_IMAGE_PIXELS = None

DEFAULT_ROOT = pathlib.Path("E:/22")
PREVIOUS_PIPELINE_SECONDS_PER_FRAME = 46.85
"""AGENTS.md §8 baseline, derived from output mtimes."""


def survey(detected: SourceSet) -> None:
    print("=== source set ===")
    print(detected.describe())
    per_frame = detected.total_bytes / max(len(detected.frames), 1)
    print(f"per frame  : {per_frame / 2**20:.1f} MiB for all {detected.camera_count} cameras")
    try:
        rig = rig_for(detected.camera_count)
    except UnknownRigError as exc:
        print(f"rig        : UNKNOWN -- {exc}")
        return
    share = len(rig.unique_indices) / rig.file_count
    print(
        f"rig        : {rig.name}, {len(rig.unique_indices)} distinct of {rig.file_count} "
        f"files -> {per_frame * share / 2**20:.1f} MiB/frame when the duplicates are "
        f"skipped ({(1 - share) * 100:.0f}% less read I/O)"
    )


def survey_previous_output(detected: SourceSet) -> None:
    print("\n=== previous pipeline output ===")
    if detected.output_dir is None or not detected.output_dir.is_dir():
        print("no output directory alongside the source")
        return
    files = sorted(detected.output_dir.glob("*.png"))
    if not files:
        print(f"{detected.output_dir} is empty")
        return
    numbers = sorted(int(path.name.split(".")[-2]) for path in files)
    gaps = set(range(numbers[0], numbers[-1] + 1)) - set(numbers)
    size = sum(path.stat().st_size for path in files)
    width, height = png_dimensions(files[0])
    print(
        f"n={len(files)} frames={numbers[0]}..{numbers[-1]} missing={len(gaps)} "
        f"{width}x{height} ({width / height:.3f}:1) {size / 2**30:.2f} GiB "
        f"{size / len(files) / 2**20:.1f} MiB/frame"
    )
    times = sorted(path.stat().st_mtime for path in files)
    span = times[-1] - times[0]
    if span <= 0:
        return
    print(
        f"wall clock : {dt.datetime.fromtimestamp(times[0]):%Y-%m-%d %H:%M} -> "
        f"{dt.datetime.fromtimestamp(times[-1]):%Y-%m-%d %H:%M} = {span / 3600:.2f} h"
    )
    print(
        f"throughput : {span / len(files):.2f} s/frame  "
        f"(recorded baseline {PREVIOUS_PIPELINE_SECONDS_PER_FRAME})"
    )


def check_alpha(detected: SourceSet, frame: int) -> None:
    print(f"\n=== alpha channel, frame {frame} ===")
    informative = False
    for camera in detected.cameras[:3]:
        with Image.open(detected.tile_path(camera.index, frame)) as image:
            if image.mode != "RGBA":
                print(f"Camera{camera.index}: mode {image.mode}, no alpha to waste")
                continue
            alpha = np.asarray(image)[:, :, 3]
        opaque = float((alpha == 255).mean())
        informative |= opaque < 1.0
        print(
            f"Camera{camera.index:<3d} min={int(alpha.min())} max={int(alpha.max())} "
            f"opaque={opaque:.4%}"
        )
    print(
        "alpha carries information"
        if informative
        else "alpha is uniformly opaque: a quarter of every file is wasted bytes"
    )


def check_duplicates(detected: SourceSet, frame: int) -> None:
    print(f"\n=== duplicate viewpoints, frame {frame} ===")
    try:
        rig = rig_for(detected.camera_count)
    except UnknownRigError:
        print("no registered rig, so there is no expected duplicate structure to confirm")
        return
    groups = [group for group in rig.groups if len(group) > 1]
    if not groups:
        print("this rig has no duplicated viewpoints")
        return
    cache: dict[int, npt.NDArray[np.int16]] = {}

    def load(index: int) -> npt.NDArray[np.int16]:
        if index not in cache:
            with Image.open(detected.tile_path(index, frame)) as image:
                cache[index] = np.asarray(image.convert("RGB"), np.int16)
        return cache[index]

    for group in groups:
        first = load(group[0])
        for other in group[1:]:
            delta = int(np.abs(first - load(other)).max())
            note = "  (render noise only)" if 0 < delta <= 2 else ""
            print(
                f"Camera{group[0]:<3d} vs Camera{other:<3d} identical={delta == 0} "
                f"max_abs_diff={delta}{note}"
            )
    print(f"{len(rig.groups)} distinct orientations across {rig.file_count} files")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, default=DEFAULT_ROOT)
    parser.add_argument("--frame", type=int, default=None, help="frame for the pixel checks")
    parser.add_argument("--skip-duplicates", action="store_true")
    args = parser.parse_args(argv)

    candidates = scan(args.source)
    usable = [candidate for candidate in candidates if candidate.usable]
    if not usable:
        print(f"no usable source set in {args.source}")
        for candidate in candidates:
            print(f"  stem {candidate.stem!r}: {'; '.join(candidate.problems)}")
        return 1
    for detected in usable:
        survey(detected)
        survey_previous_output(detected)
        frame = args.frame if args.frame is not None else detected.frames[0]
        check_alpha(detected, frame)
        if not args.skip_duplicates:
            check_duplicates(detected, frame)
    return 0


if __name__ == "__main__":
    sys.exit(main())
