"""Camera rig definitions: which orientation each source file was rendered from.

The 20-file rig here was reverse-solved from the reference data and proved end-to-end;
the method and the numbers are in AGENTS.md §3, and `tools/fit_rig.py` re-derives it.

**A rig is data, not an assumption.** The source directory is a user-chosen parameter, so
a set with a different camera count is a different rig and this module must refuse it
rather than mis-stitch it. :func:`rig_for` looks up :data:`REGISTRY` and raises
:class:`UnknownRigError` with instructions when the layout is not registered.
"""

from __future__ import annotations

import collections
import dataclasses

__all__ = [
    "REGISTRY",
    "Rig",
    "UnknownRigError",
    "View",
    "rig_for",
    "twenty_file_rig",
]


class UnknownRigError(LookupError):
    """Raised for a camera layout with no registered rig."""


@dataclasses.dataclass(frozen=True, slots=True, order=True)
class View:
    """A camera orientation in degrees. `elevation` is positive **up**."""

    yaw: float
    elevation: float

    def normalised(self) -> View:
        """Yaw folded into [0, 360) so equal orientations compare equal."""
        return View(self.yaw % 360.0, self.elevation)


@dataclasses.dataclass(frozen=True, slots=True)
class Rig:
    """A complete rig: one orientation per source file, in file-index order."""

    name: str
    views: tuple[View, ...]
    fov_deg: float = 90.0
    mirrored: bool = True
    """Image-plane handedness. True matches the production output (AGENTS.md §3)."""

    def __post_init__(self) -> None:
        if len(self.views) < 2:
            raise ValueError(f"rig {self.name!r} needs at least 2 views")
        if not 0.0 < self.fov_deg < 180.0:
            raise ValueError(f"rig {self.name!r} has an invalid fov {self.fov_deg}")

    @property
    def file_count(self) -> int:
        """Number of source files per frame, i.e. the camera directory count."""
        return len(self.views)

    def view_for(self, index: int) -> View:
        """Orientation for a 1-based camera index."""
        if not 1 <= index <= self.file_count:
            raise ValueError(f"camera index must be 1..{self.file_count}, got {index}")
        return self.views[index - 1]

    @property
    def groups(self) -> tuple[tuple[int, ...], ...]:
        """File indices grouped by orientation, ordered by first appearance.

        Derived from `views`, never hardcoded: a group with more than one member means
        the same viewpoint was rendered more than once.
        """
        buckets: dict[View, list[int]] = collections.defaultdict(list)
        for index, view in enumerate(self.views, start=1):
            buckets[view.normalised()].append(index)
        return tuple(tuple(v) for v in buckets.values())

    @property
    def unique_indices(self) -> tuple[int, ...]:
        """One representative file index per distinct orientation, ascending.

        Reading only these is what saves 25% of the input I/O on the 20-file rig.
        """
        return tuple(sorted(group[0] for group in self.groups))

    @property
    def duplicate_indices(self) -> tuple[int, ...]:
        """File indices that duplicate an orientation already covered, ascending."""
        redundant = set(range(1, self.file_count + 1)) - set(self.unique_indices)
        return tuple(sorted(redundant))

    @property
    def unique_views(self) -> dict[int, View]:
        """``{representative file index: orientation}`` for the distinct orientations."""
        return {index: self.views[index - 1] for index in self.unique_indices}


def twenty_file_rig() -> Rig:
    """The rig behind ``E:\\22``: 5 azimuth sectors x 3 elevations, written as 20 files.

    Per sector of four files: elevation +45, -45, 0, then the *next* sector's +45 again.
    That fourth slot is why 20 files carry only 15 distinct viewpoints.
    """
    sectors, yaw_step = 5, 72.0
    views: list[View] = []
    for sector in range(sectors):
        yaw = yaw_step * sector
        views += [View(yaw, 45.0), View(yaw, -45.0), View(yaw, 0.0)]
        views.append(View(yaw_step * ((sector + 1) % sectors), 45.0))
    return Rig(name="of3d-20", views=tuple(views), fov_deg=90.0, mirrored=True)


REGISTRY: dict[int, Rig] = {20: twenty_file_rig()}
"""Registered rigs by source-file count. Only the 20-file layout is known so far."""


def rig_for(file_count: int) -> Rig:
    """The registered rig for this many source files.

    Raises :class:`UnknownRigError` for anything unregistered -- guessing would produce a
    panorama that looks fine on a thumbnail and is geometrically wrong.
    """
    try:
        return REGISTRY[file_count]
    except KeyError:
        known = ", ".join(str(k) for k in sorted(REGISTRY))
        raise UnknownRigError(
            f"no rig registered for {file_count} cameras (registered: {known}). "
            "Solve the layout with `tools/fit_rig.py` against a reference equirect, "
            "then add it to vr_compose.rig.REGISTRY."
        ) from None
