"""Copy the first N frames of every camera out of a source directory, structure intact.

For carrying a sample of a render from one machine to another -- a few hundred MB instead
of the whole sequence -- so it can be analysed (`tools/rig_probe.py`, `vr-compose frame`)
where the tooling is. The destination is a valid source directory in its own right: the
same `CameraN/[inner]/<stem>.<frame>.png` layout, so `vr-compose` reads it unchanged.

**Standard library only, Python 3.9+**, so it runs on a machine that has nothing else
installed. It does not import `vr_compose`.

    python copy_frames.py D:/CR/0909C E:/sample                 # first 100 frames, every camera
    python copy_frames.py D:/CR/0909C E:/sample --count 5       # a quick sample
    python copy_frames.py D:/CR/0909C E:/sample --start 1656 --count 100
    python copy_frames.py D:/CR/0909C E:/sample --common        # only frames every camera has
    python copy_frames.py D:/CR/0909C E:/sample --dry-run       # list, copy nothing

"First N" is per camera by default: each camera's lowest N frame numbers (from `--start`
if given). Cameras that hold different frames are reported, because a set where they
differ will be refused by `vr-compose` -- pass `--common` to copy only the frames every
camera has, or fix the render. Existing destination files of the same size are skipped, so
an interrupted copy can be re-run.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import pathlib
import re
import shutil
import sys
import time

CAMERA_RE = re.compile(r"^camera[ _-]?(\d+)$", re.IGNORECASE)
FRAME_RE = re.compile(r"^(?P<stem>.+?)\.(?P<frame>\d+)\.png$", re.IGNORECASE)


def tile_directory(camera_dir: pathlib.Path) -> pathlib.Path | None:
    """The camera directory itself, or the one subdirectory holding numbered PNGs."""
    if any(FRAME_RE.match(p.name) for p in camera_dir.iterdir() if p.is_file()):
        return camera_dir
    for child in sorted(p for p in camera_dir.iterdir() if p.is_dir()):
        if any(FRAME_RE.match(p.name) for p in child.iterdir() if p.is_file()):
            return child
    return None


def scan(
    root: pathlib.Path, stem: str | None
) -> dict[int, tuple[pathlib.Path, dict[int, pathlib.Path]]]:
    """`{camera index: (tile directory, {frame: path})}` for the chosen stem."""
    cameras: dict[int, tuple[pathlib.Path, dict[int, pathlib.Path]]] = {}
    stems_seen: dict[str, int] = {}
    for entry in sorted(root.iterdir()):
        match = CAMERA_RE.match(entry.name) if entry.is_dir() else None
        if match is None:
            continue
        index = int(match.group(1))
        tile_dir = tile_directory(entry)
        if tile_dir is None:
            print(f"  {entry.name}: no numbered PNGs, skipped", file=sys.stderr)
            continue
        frames: dict[int, pathlib.Path] = {}
        for path in tile_dir.iterdir():
            m = FRAME_RE.match(path.name)
            if m is None or not path.is_file():
                continue
            stems_seen[m.group("stem")] = stems_seen.get(m.group("stem"), 0) + 1
            if stem is not None and m.group("stem") != stem:
                continue
            frames[int(m.group("frame"))] = path
        cameras[index] = (tile_dir, frames)
    if stem is None and len(stems_seen) > 1:
        names = ", ".join(f"{s} ({n})" for s, n in sorted(stems_seen.items()))
        raise SystemExit(f"several stems under {root}: {names}. Choose one with --stem.")
    return cameras


def choose(frames: dict[int, pathlib.Path], start: int | None, count: int) -> list[int]:
    ordered = sorted(f for f in frames if start is None or f >= start)
    return ordered[:count]


def copy_one(src: pathlib.Path, dst: pathlib.Path) -> tuple[int, bool]:
    """`(bytes, copied)`; an existing file of the same size is left alone."""
    size = src.stat().st_size
    if dst.exists() and dst.stat().st_size == size:
        return size, False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return size, True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", type=pathlib.Path, help="directory holding Camera1..CameraN")
    parser.add_argument("destination", type=pathlib.Path, help="where the sample goes")
    parser.add_argument("--count", type=int, default=100, help="frames per camera (default 100)")
    parser.add_argument("--start", type=int, default=None, help="first frame number to take")
    parser.add_argument("--stem", default=None, help="which render, when a camera holds several")
    parser.add_argument("--common", action="store_true", help="only frames present in every camera")
    parser.add_argument("--workers", type=int, default=4, help="parallel copies")
    parser.add_argument("--dry-run", action="store_true", help="list what would be copied")
    args = parser.parse_args(argv)

    if not args.source.is_dir():
        raise SystemExit(f"{args.source} is not a directory")
    if args.count < 1:
        raise SystemExit("--count must be at least 1")
    cameras = scan(args.source, args.stem)
    if len(cameras) < 2:
        raise SystemExit(f"fewer than two Camera* directories with PNGs under {args.source}")

    chosen: dict[int, list[int]] = {}
    if args.common:
        shared = None
        for _index, (_dir, frames) in cameras.items():
            shared = set(frames) if shared is None else shared & set(frames)
        common = choose({f: pathlib.Path() for f in (shared or set())}, args.start, args.count)
        chosen = {index: common for index in cameras}
    else:
        chosen = {
            index: choose(frames, args.start, args.count) for index, (_d, frames) in cameras.items()
        }

    print(f"source      : {args.source}")
    print(f"destination : {args.destination}")
    for index in sorted(cameras):
        tile_dir, frames = cameras[index]
        picked = chosen[index]
        span = f"{picked[0]}..{picked[-1]}" if picked else "none"
        available = f"{len(frames):5d} frame(s) available"
        print(f"  Camera{index:<3} {available}, taking {len(picked):4d} ({span})")

    distinct = {tuple(v) for v in chosen.values()}
    if len(distinct) > 1:
        print(
            "\nWARNING: the cameras do not hold the same frames, so the copy will not be a set "
            "vr-compose accepts as-is. Re-run with --common, or check the render.",
            file=sys.stderr,
        )

    jobs: list[tuple[pathlib.Path, pathlib.Path]] = []
    for index in sorted(cameras):
        tile_dir, frames = cameras[index]
        relative = tile_dir.relative_to(args.source)
        for frame in chosen[index]:
            src = frames[frame]
            jobs.append((src, args.destination / relative / src.name))
    total = len(jobs)
    if args.dry_run:
        print(f"\ndry run: {total} file(s) would be copied")
        return 0
    if not total:
        print("nothing to copy")
        return 1

    started = time.time()
    copied = skipped = 0
    moved = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for done, (size, did_copy) in enumerate(pool.map(lambda j: copy_one(*j), jobs), 1):
            moved += size if did_copy else 0
            copied += did_copy
            skipped += not did_copy
            if done % 20 == 0 or done == total:
                elapsed = time.time() - started
                rate = moved / 2**20 / elapsed if elapsed else 0.0
                print(
                    f"\r  {done}/{total}  {moved / 2**30:.2f} GiB  {rate:.0f} MiB/s",
                    end="",
                    flush=True,
                )
    print()
    print(
        f"done: {copied} copied, {skipped} already present, {moved / 2**30:.2f} GiB in "
        f"{time.time() - started:.0f} s -> {args.destination}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
