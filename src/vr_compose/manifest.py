"""The rig manifest VROpPanoramic writes beside its tiles, and the rig it decides.

Upstream's H1: every capture writes ``vrop_panoramic_manifest.json`` in its output
directory, beside ``Camera01..Camera15``, recording the rig and settings it actually
applied. Its contract fields are `schema`, `schema_version`, `state`, `rig` (type,
coordinate system, orientation reference, and per view the directory, pitch, yaw, roll,
field of view and the frames written), `tiles` and `frames.rate`; a change of meaning
bumps the version, an added field does not.

A 15-directory capture and a 20-directory Camera360 capture with only ``Camera1..15``
copied look the same on disk (ROADMAP P11). The manifest is how a capture says which it
is, so a usable one builds the rig and the layout gate does not run. The user's rule
(2026-11-03), in :func:`choose_rig`'s three outcomes:

* the manifest is usable -- a schema this module reads, state `finished`, the stem being
  read in `tiles.stems`, and a description of exactly the files on disk -- and its views
  are a registered rig: that rig, in the manifest's directory order, with no layout check;
* the manifest is usable but its views are no registered rig (a six-view Cubemap, a
  rolled or wider view): :class:`~vr_compose.rig.UnknownRigError`, as for an unknown
  directory count;
* anything else -- no manifest, one this module cannot read, an unfinished capture,
  another stem's, or one describing other files: the registry and the layout gate, as
  without a manifest, and the report says why.
"""

from __future__ import annotations

import dataclasses
import fractions
import json
import pathlib
from collections.abc import Mapping

from vr_compose.rig import ANGLE_DECIMALS, Rig, UnknownRigError, View, rig_for, rig_for_views
from vr_compose.source import CameraFiles, SourceSet

__all__ = [
    "MANIFEST_NAME",
    "SCHEMA",
    "SCHEMA_VERSIONS",
    "Manifest",
    "ManifestError",
    "ManifestView",
    "RigChoice",
    "Written",
    "choose_rig",
    "describe_rate",
    "read",
]

MANIFEST_NAME = "vrop_panoramic_manifest.json"
SCHEMA = "vrop-panoramic-manifest"
SCHEMA_VERSIONS = (1,)
"""The versions this module reads. A higher one changed a field's meaning."""
COORDINATE_SYSTEMS = ("unreal",)
"""`unreal`: degrees, pitch positive up, yaw positive to the right seen from above, roll 0.
That is :class:`~vr_compose.rig.View` with yaw = UE yaw and elevation = UE pitch, in the
registered rigs' handedness (`mirrored`, AGENTS.md section 3)."""
ORIENTATION_REFERENCES = ("world", "camera")
"""Both stitch the same way: every view of a frame shares the reference."""
ENCODINGS = {"png": "srgb8", "exr": "linear_half"}
"""How :mod:`vr_compose.io` reads each tile format; a manifest saying otherwise describes
pixels this program would misread."""


class ManifestError(ValueError):
    """A manifest that cannot decide the rig for this source; the message says why."""


@dataclasses.dataclass(frozen=True, slots=True)
class Written:
    """The frames one view's directory received: how many, first and last frame number."""

    count: int
    first: int
    last: int


@dataclasses.dataclass(frozen=True, slots=True)
class ManifestView:
    """One view as the manifest records it, in degrees in its coordinate system."""

    directory: str
    pitch: float
    yaw: float
    roll: float
    fov: float
    written: Written | None
    """Absent until the capture has finished writing."""


@dataclasses.dataclass(frozen=True, slots=True)
class Manifest:
    """The contract fields of one manifest."""

    path: pathlib.Path
    schema_version: int
    state: str
    rig_type: str
    coordinate_system: str
    orientation_reference: str
    views: tuple[ManifestView, ...]
    tile_width: int
    tile_height: int
    tile_format: str
    encoding: str
    alpha: str
    stems: tuple[str, ...]
    rate: fractions.Fraction


def _get(obj: Mapping[str, object], key: str, where: str) -> object:
    try:
        return obj[key]
    except KeyError:
        raise ManifestError(f"{where}{key} is missing") from None


def _object(value: object, where: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ManifestError(f"{where.rstrip('.') or 'the file'} is not a JSON object")
    return value


def _text(obj: Mapping[str, object], key: str, where: str) -> str:
    value = _get(obj, key, where)
    if not isinstance(value, str):
        raise ManifestError(f"{where}{key} is not a string")
    return value


def _choice(obj: Mapping[str, object], key: str, where: str, choices: tuple[str, ...]) -> str:
    value = _text(obj, key, where)
    if value not in choices:
        raise ManifestError(f"{where}{key} {value!r} is not one this VR-Compose reads")
    return value


def _integer(obj: Mapping[str, object], key: str, where: str) -> int:
    value = _get(obj, key, where)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{where}{key} is not an integer")
    return value


def _number(obj: Mapping[str, object], key: str, where: str) -> float:
    value = _get(obj, key, where)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ManifestError(f"{where}{key} is not a number")
    return float(value)


def _view(value: object, where: str) -> ManifestView:
    obj = _object(value, where)
    written: Written | None = None
    if "written" in obj:
        record = _object(obj["written"], f"{where}written.")
        written = Written(
            count=_integer(record, "count", f"{where}written."),
            first=_integer(record, "first", f"{where}written."),
            last=_integer(record, "last", f"{where}written."),
        )
    return ManifestView(
        directory=_text(obj, "directory", where),
        pitch=_number(obj, "pitch", where),
        yaw=_number(obj, "yaw", where),
        roll=_number(obj, "roll", where),
        fov=_number(obj, "fov", where),
        written=written,
    )


def read(path: pathlib.Path) -> Manifest:
    """The manifest at `path`. Raises :class:`ManifestError` for anything that is not a
    manifest of a schema version this module reads, with every contract field present and
    of its type. The state is read, not judged: an unfinished capture's manifest parses."""
    try:
        document: object = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError) as exc:
        raise ManifestError(f"cannot be read ({exc})") from None
    except json.JSONDecodeError as exc:
        raise ManifestError(f"is not JSON ({exc})") from None
    top = _object(document, "")
    schema = _text(top, "schema", "")
    if schema != SCHEMA:
        raise ManifestError(f"schema {schema!r} is not {SCHEMA!r}")
    version = _integer(top, "schema_version", "")
    if version not in SCHEMA_VERSIONS:
        known = ", ".join(str(v) for v in SCHEMA_VERSIONS)
        raise ManifestError(f"schema version {version}; this VR-Compose reads {known}")

    rig = _object(_get(top, "rig", ""), "rig.")
    views = _get(rig, "views", "rig.")
    if not isinstance(views, list) or not views:
        raise ManifestError("rig.views is not a non-empty list")
    tiles = _object(_get(top, "tiles", ""), "tiles.")
    stems = _get(tiles, "stems", "tiles.")
    if not isinstance(stems, list) or not all(isinstance(stem, str) for stem in stems):
        raise ManifestError("tiles.stems is not a list of strings")
    frames = _object(_get(top, "frames", ""), "frames.")
    rate = _object(_get(frames, "rate", "frames."), "frames.rate.")
    numerator = _integer(rate, "numerator", "frames.rate.")
    denominator = _integer(rate, "denominator", "frames.rate.")
    if numerator <= 0 or denominator <= 0:
        raise ManifestError(f"frames.rate {numerator}/{denominator} is not a positive rate")
    return Manifest(
        path=path,
        schema_version=version,
        state=_text(top, "state", ""),
        rig_type=_text(rig, "type", "rig."),
        coordinate_system=_choice(rig, "coordinate_system", "rig.", COORDINATE_SYSTEMS),
        orientation_reference=_choice(rig, "orientation_reference", "rig.", ORIENTATION_REFERENCES),
        views=tuple(_view(view, f"rig.views[{i}].") for i, view in enumerate(views)),
        tile_width=_integer(tiles, "width", "tiles."),
        tile_height=_integer(tiles, "height", "tiles."),
        tile_format=_text(tiles, "format", "tiles."),
        encoding=_choice(tiles, "encoding", "tiles.", tuple(ENCODINGS.values())),
        alpha=_text(tiles, "alpha", "tiles."),
        stems=tuple(stems),
        rate=fractions.Fraction(numerator, denominator),
    )


def describe_rate(rate: fractions.Fraction) -> str:
    """``30``, ``60``, or ``29.97`` for 30000/1001."""
    return f"{float(rate):g}"


@dataclasses.dataclass(frozen=True, slots=True)
class RigChoice:
    """The rig a source is stitched with, and what its manifest had to do with it."""

    rig: Rig
    manifest: Manifest | None = None
    """The manifest that built `rig`; None when the registry did."""
    set_aside: str | None = None
    """Why a manifest that is present did not build the rig."""

    @property
    def rate(self) -> fractions.Fraction | None:
        """The capture's frame rate, from the manifest that built the rig."""
        return self.manifest.rate if self.manifest is not None else None

    def lines(self) -> list[str]:
        """What a report says about the manifest, under its `rig` line."""
        if self.manifest is None:
            state = "none" if self.set_aside is None else f"not used -- {self.set_aside}"
            return [f"manifest   : {state}"]
        found = self.manifest
        return [
            f"manifest   : {found.path.name} -- schema {found.schema_version}, {found.state}, "
            f"{found.rig_type}, {len(found.views)} views relative to the "
            f"{found.orientation_reference}, {describe_rate(found.rate)} fps",
            "layout     : from manifest -- the layout gate is not run",
        ]


def _directory_name(source: SourceSet, camera: CameraFiles) -> str:
    """The ``CameraNN`` directory under the root, whether the tiles are in it or below it."""
    return camera.directory.relative_to(source.root).parts[0]


def _frames(count: int, first: int, last: int) -> str:
    return f"{count} frame(s) {first}..{last}"


def _cameras_for(found: Manifest, source: SourceSet) -> list[CameraFiles]:
    """The disk's camera for each manifest view, in the manifest's order. Raises
    :class:`ManifestError` when the manifest does not describe exactly these files."""
    if found.state != "finished":
        why = {
            "rendering": "the capture did not finish",
            "failed": "Movie Render Queue ended the job with an error",
        }.get(found.state, "not a finished capture")
        raise ManifestError(f"state {found.state!r}: {why}")
    if source.stem not in found.stems:
        listed = ", ".join(repr(stem) for stem in found.stems) or "none"
        raise ManifestError(f"it lists stems {listed}, not {source.stem!r}")
    if found.tile_format != source.tile_format:
        raise ManifestError(
            f"it describes {found.tile_format} tiles; the disk holds {source.tile_format}"
        )
    expected = ENCODINGS.get(found.tile_format)
    if found.encoding != expected:
        raise ManifestError(
            f"it describes {found.tile_format} tiles encoded as {found.encoding}; "
            f"VR-Compose reads {found.tile_format} as {expected}"
        )
    size = source.tile_size
    if (found.tile_width, found.tile_height) != (size, size):
        raise ManifestError(
            f"it describes {found.tile_width}x{found.tile_height} tiles; "
            f"the disk holds {size}x{size}"
        )

    on_disk = {_directory_name(source, camera).lower(): camera for camera in source.cameras}
    named = [view.directory.lower() for view in found.views]
    repeated = sorted({v.directory for v in found.views if named.count(v.directory.lower()) > 1})
    if repeated:
        raise ManifestError(f"it names {', '.join(repeated)} more than once")
    missing = [view.directory for view in found.views if view.directory.lower() not in on_disk]
    extra = [_directory_name(source, cam) for name, cam in on_disk.items() if name not in named]
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"{', '.join(missing)} in it but not on disk")
        if extra:
            parts.append(f"{', '.join(extra)} on disk but not in it")
        raise ManifestError(f"its directories are not the disk's: {'; '.join(parts)}")

    cameras = [on_disk[view.directory.lower()] for view in found.views]
    for view, camera in zip(found.views, cameras, strict=True):
        written = view.written
        if written is None:
            raise ManifestError(f"it records no frames written for {view.directory}")
        frames = camera.frames
        disk = _frames(len(frames), frames[0], frames[-1])
        if len(found.stems) == 1:
            agrees = (written.count, written.first, written.last) == (
                len(frames), frames[0], frames[-1],
            )  # fmt: skip
        else:
            # The record covers every stem; this one's frames can only be inside it.
            agrees = written.first <= frames[0] and frames[-1] <= written.last
        if not agrees:
            raise ManifestError(
                f"{view.directory}: it records "
                f"{_frames(written.count, written.first, written.last)} written, the disk "
                f"holds {disk} -- the files are not (only) this capture's"
            )
    return cameras


def _rig_from(found: Manifest, cameras: list[CameraFiles]) -> Rig:
    """The registered rig the manifest's views are, ordered by camera index."""
    rolled = [v.directory for v in found.views if round(v.roll, ANGLE_DECIMALS) != 0.0]
    if rolled:
        raise UnknownRigError(
            f"the manifest describes rolled views ({', '.join(rolled)}); no registered rig has roll"
        )
    fovs = sorted({round(view.fov, ANGLE_DECIMALS) for view in found.views})
    if len(fovs) != 1:
        raise UnknownRigError(
            f"the manifest describes views of different fields of view ({fovs}); a "
            "registered rig has one"
        )
    by_index = sorted(zip(cameras, found.views, strict=True), key=lambda pair: pair[0].index)
    views = [View(view.yaw, view.pitch) for _, view in by_index]
    try:
        rig = rig_for_views(views, fovs[0])
    except UnknownRigError as exc:
        raise UnknownRigError(f"the manifest's {found.rig_type} rig: {exc}") from None
    return dataclasses.replace(rig, needs_layout_check=False)


def choose_rig(source: SourceSet) -> RigChoice:
    """The rig to stitch `source` with: its manifest's when that is usable, otherwise the
    registry's by directory count -- see the module docstring.

    Raises :class:`~vr_compose.rig.UnknownRigError` when a usable manifest describes a
    rig that is not registered, and when the registry has none for the directory count;
    the latter message says why a manifest that was present was not used.
    """
    path = source.root / MANIFEST_NAME
    if not path.is_file():
        return RigChoice(rig_for(source.camera_count))
    try:
        found = read(path)
        cameras = _cameras_for(found, source)
    except ManifestError as exc:
        try:
            return RigChoice(rig_for(source.camera_count), set_aside=str(exc))
        except UnknownRigError as unknown:
            raise UnknownRigError(f"{unknown} The manifest was not used: {exc}") from None
    return RigChoice(_rig_from(found, cameras), manifest=found)
