"""Equirectangular <-> pinhole geometry. Pure maths, no I/O, no rig knowledge.

Conventions, all verified against the production output (AGENTS.md §3):

World is right-handed with **Z up**. A direction is built from longitude and latitude::

    d(lon, lat) = (cos(lat)cos(lon), cos(lat)sin(lon), sin(lat))

Equirectangular mapping, width ``W`` and height ``H``::

    u = (lon / 2pi + 0.5) * W      lon in [-pi, pi), so the left edge is -pi
    v = (0.5 - lat / pi) * H       lat = +pi/2 at the top row

Camera space looks along **+X**; image +x maps to +Y and image +y (downward) to -Z.
That last part is the handedness the renderer used, and getting it wrong produces a
panorama that looks plausible at a glance because the sphere is symmetric -- which is
why `vr_compose.verify` exists.

`elevation` is positive **up**, hence the negated angle in :func:`camera_to_world`.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
I32 = npt.NDArray[np.int32]
Bool = npt.NDArray[np.bool_]

__all__ = [
    "camera_to_world",
    "equirect_directions",
    "half_extent",
    "project_to_tile",
    "rot_y",
    "rot_z",
    "tile_pixel_index",
    "tile_rays",
]


def half_extent(fov_deg: float) -> float:
    """``tan(fov/2)``: the half-width of the image plane at unit distance.

    For the 90 degree square FOV this project's rig uses, it is exactly 1.0, which makes
    a tile geometrically a cube face.
    """
    if not 0.0 < fov_deg < 180.0:
        raise ValueError(f"fov must be in (0, 180) degrees, got {fov_deg}")
    return float(np.tan(np.radians(fov_deg) / 2.0))


def rot_z(deg: float) -> F64:
    """Rotation about the up axis (yaw)."""
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(deg: float) -> F64:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def camera_to_world(yaw_deg: float, elevation_deg: float) -> F64:
    """Camera-space -> world-space rotation.

    The negated elevation is deliberate: positive elevation means looking **up**, while
    :func:`rot_y` with a positive angle tilts the +X axis downward.
    """
    return rot_z(yaw_deg) @ rot_y(-elevation_deg)


def equirect_directions(width: int, height: int, *, rows: slice | None = None) -> F64:
    """Unit world directions for equirect pixel centres, shape ``(3, n)`` row-major.

    `rows` restricts the computation to a horizontal band, which is how callers keep
    peak memory bounded at 8K -- the full 7680x3840 direction array is 708 MB in float64.
    """
    if width < 2 or height < 2:
        raise ValueError(f"equirect must be at least 2x2, got {width}x{height}")
    row_index = np.arange(height, dtype=np.float64)[rows if rows is not None else slice(None)]
    lon = ((np.arange(width, dtype=np.float64) + 0.5) / width - 0.5) * 2.0 * np.pi
    lat = (0.5 - (row_index + 0.5) / height) * np.pi
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    cos_lat = np.cos(lat_grid)
    dirs = np.stack([cos_lat * np.cos(lon_grid), cos_lat * np.sin(lon_grid), np.sin(lat_grid)])
    return np.asarray(dirs.reshape(3, -1), dtype=np.float64)


def tile_rays(size: int, fov_deg: float, *, mirrored: bool = True) -> F64:
    """Unit camera-space directions for an ``size x size`` tile, shape ``(3, size**2)``.

    `mirrored=True` is the handedness that matches the production output; the flipped
    variant exists only so :mod:`vr_compose.verify` can demonstrate which one is right.
    """
    if size < 1:
        raise ValueError(f"tile size must be positive, got {size}")
    t = half_extent(fov_deg)
    grid = (np.arange(size, dtype=np.float64) + 0.5) / size * 2.0 - 1.0
    x, y = np.meshgrid(grid, grid)
    sign = 1.0 if mirrored else -1.0
    dirs = np.stack([np.ones_like(x), sign * x * t, -y * t]).reshape(3, -1)
    return np.asarray(dirs / np.linalg.norm(dirs, axis=0), dtype=np.float64)


def project_to_tile(
    dirs: F64, yaw_deg: float, elevation_deg: float, fov_deg: float, *, mirrored: bool = True
) -> tuple[F64, F64, Bool]:
    """Project world directions into one tile's image plane.

    Returns ``(x, y, visible)``. `x` and `y` are normalised image-plane coordinates in
    ``[-t, t]`` with ``t = tan(fov/2)``; `visible` marks the directions inside the
    frustum. Coordinates outside the frustum are returned unclamped and must not be used.
    """
    t = half_extent(fov_deg)
    cam = camera_to_world(yaw_deg, elevation_deg).T @ dirs
    forward = cam[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = cam[1] / forward
        y = -cam[2] / forward
    if not mirrored:
        x = -x
    visible = (forward > 0.0) & (np.abs(x) <= t) & (np.abs(y) <= t)
    return (
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(visible, dtype=np.bool_),
    )


def tile_pixel_index(x: F64, y: F64, size: int, fov_deg: float) -> tuple[I32, I32]:
    """Normalised image-plane coordinates -> nearest pixel ``(column, row)``.

    Nearest neighbour is the P1 sampler. P4 replaces it with an anisotropic filter; the
    interface stays because the LUT stores the same coordinates either way.
    """
    t = half_extent(fov_deg)
    column = np.clip(((x / t + 1.0) * 0.5 * size).astype(np.int32), 0, size - 1)
    row = np.clip(((y / t + 1.0) * 0.5 * size).astype(np.int32), 0, size - 1)
    return np.asarray(column, np.int32), np.asarray(row, np.int32)
