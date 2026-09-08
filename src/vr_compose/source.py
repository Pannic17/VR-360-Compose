"""Locate and validate a source frame set by its directory shape.

The source directory is a user-chosen parameter and, when not given, is found next to the
executable. So nothing about the reference data may be assumed: not the camera count, not
the inner directory name, not the file stem, not the frame numbering. All of it is
inferred, and anything inconsistent is reported instead of worked around -- a set with
ragged frame counts or non-square tiles is a different kind of data, not a subset of this
one. See AGENTS.md §6.
"""

from __future__ import annotations

import collections
import dataclasses
import pathlib
import re
import struct
import sys

__all__ = [
    "CameraFiles",
    "SourceSet",
    "candidate_roots",
    "discover",
    "png_dimensions",
    "scan",
]

CAMERA_RE = re.compile(r"^camera[\s_-]*(\d+)$", re.IGNORECASE)
FRAME_RE = re.compile(r"^(?P<stem>.+)\.(?P<frame>\d{2,8})\.png$", re.IGNORECASE)
OUTPUT_DIR_NAMES = frozenset({"finishtaskoutput", "output", "out"})
TILE_SEARCH_DEPTH = 1
"""How far below ``CameraN/`` to look for tiles. The reference data uses ``Tempory/``."""

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def png_dimensions(path: pathlib.Path) -> tuple[int, int]:
    """``(width, height)`` from the IHDR chunk. Reads 24 bytes, never decodes."""
    with open(path, "rb") as handle:
        head = handle.read(24)
    if head[:8] != _PNG_MAGIC:
        raise ValueError(f"not a PNG: {path}")
    width, height = struct.unpack(">II", head[16:24])
    return int(width), int(height)


@dataclasses.dataclass(frozen=True, slots=True)
class CameraFiles:
    """One camera's frames for one stem."""

    index: int
    directory: pathlib.Path
    frames: tuple[int, ...]
    width: int
    height: int
    total_bytes: int

    def path_for(self, stem: str, frame: int, digits: int) -> pathlib.Path:
        return self.directory / f"{stem}.{frame:0{digits}d}.png"


@dataclasses.dataclass(frozen=True, slots=True)
class SourceSet:
    """One stem (one scene) across all its cameras, plus whatever does not add up."""

    root: pathlib.Path
    stem: str
    cameras: tuple[CameraFiles, ...]
    frame_digits: int
    output_dir: pathlib.Path | None
    problems: tuple[str, ...]

    @property
    def camera_count(self) -> int:
        return len(self.cameras)

    @property
    def frames(self) -> tuple[int, ...]:
        """Frames every camera has. Only these can be processed."""
        if not self.cameras:
            return ()
        shared = set(self.cameras[0].frames)
        for camera in self.cameras[1:]:
            shared &= set(camera.frames)
        return tuple(sorted(shared))

    @property
    def tile_size(self) -> int | None:
        """Tile edge length, or None if the cameras disagree or tiles are not square."""
        sizes = {(c.width, c.height) for c in self.cameras}
        if len(sizes) != 1:
            return None
        width, height = sizes.pop()
        return width if width == height else None

    @property
    def total_bytes(self) -> int:
        return sum(c.total_bytes for c in self.cameras)

    @property
    def usable(self) -> bool:
        return self.camera_count >= 2 and bool(self.frames) and not self.problems

    def tile_path(self, camera_index: int, frame: int) -> pathlib.Path:
        """Path of one tile. `camera_index` is 1-based."""
        for camera in self.cameras:
            if camera.index == camera_index:
                return camera.path_for(self.stem, frame, self.frame_digits)
        raise ValueError(f"no camera {camera_index} in {self.root}")

    def describe(self) -> str:
        frames = self.frames
        size = self.tile_size
        span = f"{frames[0]}..{frames[-1]} ({len(frames)} present)" if frames else "none shared"
        lines = [
            f"root       : {self.root}",
            f"stem       : {self.stem}",
            f"cameras    : {self.camera_count}",
            f"tile       : {f'{size}x{size}' if size else 'INCONSISTENT / not square'}",
            f"frames     : {span}",
            f"size       : {self.total_bytes / 2**30:.2f} GiB",
            f"output dir : {self.output_dir if self.output_dir else '(none)'}",
        ]
        lines += [f"PROBLEM    : {p}" for p in self.problems]
        return "\n".join(lines)


def _tile_directory(camera_dir: pathlib.Path) -> pathlib.Path | None:
    """Where a camera's PNGs live: the directory itself, or one level below."""
    queue: list[tuple[pathlib.Path, int]] = [(camera_dir, 0)]
    while queue:
        directory, depth = queue.pop(0)
        try:
            entries = list(directory.iterdir())
        except OSError:
            continue
        if any(entry.is_file() and FRAME_RE.match(entry.name) for entry in entries):
            return directory
        if depth < TILE_SEARCH_DEPTH:
            queue += [(e, depth + 1) for e in entries if e.is_dir()]
    return None


def scan(root: pathlib.Path) -> list[SourceSet]:
    """Every source set directly under `root`, one per file stem, sorted by stem.

    A stem is whatever UE named the render. Several stems in one root just mean several
    scenes or versions, and the prefix is never special-cased.
    """
    if not root.is_dir():
        return []
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return []

    camera_dirs: list[tuple[int, pathlib.Path]] = []
    output_dir: pathlib.Path | None = None
    for entry in entries:
        if not entry.is_dir():
            continue
        if (match := CAMERA_RE.match(entry.name)) is not None:
            camera_dirs.append((int(match.group(1)), entry))
        elif entry.name.lower() in OUTPUT_DIR_NAMES:
            output_dir = entry
    if len(camera_dirs) < 2:
        return []

    by_stem: dict[str, list[CameraFiles]] = collections.defaultdict(list)
    digits_by_stem: dict[str, set[int]] = collections.defaultdict(set)
    shared_problems: list[str] = []
    for index, camera_dir in sorted(camera_dirs):
        tile_dir = _tile_directory(camera_dir)
        if tile_dir is None:
            shared_problems.append(f"Camera{index}: no numbered PNGs found")
            continue
        grouped: dict[str, list[pathlib.Path]] = collections.defaultdict(list)
        for path in tile_dir.iterdir():
            if path.is_file() and (match := FRAME_RE.match(path.name)):
                grouped[match.group("stem")].append(path)
                digits_by_stem[match.group("stem")].add(len(match.group("frame")))
        for stem, paths in grouped.items():
            frames = sorted(int(m.group("frame")) for p in paths if (m := FRAME_RE.match(p.name)))
            try:
                width, height = png_dimensions(paths[0])
            except (OSError, ValueError) as exc:
                shared_problems.append(f"Camera{index}/{stem}: {exc}")
                continue
            by_stem[stem].append(
                CameraFiles(
                    index=index,
                    directory=tile_dir,
                    frames=tuple(frames),
                    width=width,
                    height=height,
                    total_bytes=sum(p.stat().st_size for p in paths),
                )
            )

    results: list[SourceSet] = []
    for stem, cameras in sorted(by_stem.items()):
        cameras.sort(key=lambda camera: camera.index)
        results.append(
            SourceSet(
                root=root,
                stem=stem,
                cameras=tuple(cameras),
                frame_digits=max(digits_by_stem[stem], default=4),
                output_dir=output_dir,
                problems=tuple(_problems(cameras, shared_problems, digits_by_stem[stem])),
            )
        )
    return results


def _problems(cameras: list[CameraFiles], shared: list[str], digit_widths: set[int]) -> list[str]:
    found = list(shared)
    indices = [camera.index for camera in cameras]
    if indices != list(range(1, len(indices) + 1)):
        found.append(f"camera numbering is not 1..{len(indices)}: {indices}")
    resolutions = {(camera.width, camera.height) for camera in cameras}
    if len(resolutions) != 1:
        found.append(f"cameras disagree on tile resolution: {sorted(resolutions)}")
    else:
        width, height = next(iter(resolutions))
        if width != height:
            found.append(
                f"tiles are not square ({width}x{height}); the registered rigs assume a "
                "square FOV, so this is a different layout"
            )
    counts = {len(camera.frames) for camera in cameras}
    if len(counts) != 1:
        found.append(f"cameras hold different frame counts: {sorted(counts)}")
    if len(digit_widths) > 1:
        found.append(f"frame numbers are zero-padded inconsistently: {sorted(digit_widths)}")
    return found


def candidate_roots(explicit: pathlib.Path | None = None) -> list[pathlib.Path]:
    """Directories to try when no source was given, in order.

    Frozen by PyInstaller, `sys.executable` is the application itself, so "next to the
    exe" is its own directory; from source it is the project root. Either way: that
    directory, its immediate subdirectories, then the parent's subdirectories.
    """
    if explicit is not None:
        return [explicit]
    if getattr(sys, "frozen", False):
        base = pathlib.Path(sys.executable).parent
    else:
        base = pathlib.Path(__file__).resolve().parents[2]
    roots = [base]
    for parent in (base, base.parent):
        try:
            roots += sorted(path for path in parent.iterdir() if path.is_dir())
        except OSError:
            continue
    seen: set[pathlib.Path] = set()
    unique: list[pathlib.Path] = []
    for path in roots:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def discover(
    explicit: pathlib.Path | None = None,
) -> tuple[list[SourceSet], list[pathlib.Path]]:
    """``(usable source sets, directories searched)``.

    Stops at the first root that yields anything usable. More than one set means several
    scenes; the caller must choose, because silently taking the first match would process
    the wrong one.
    """
    searched: list[pathlib.Path] = []
    for root in candidate_roots(explicit):
        searched.append(root)
        found = [source for source in scan(root) if source.usable]
        if found:
            return found, searched
    return [], searched
