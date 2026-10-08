"""Rig definitions and the registry.

The 20-file layout's most useful property -- that 20 files hold only 15 distinct
viewpoints -- is *derived* from the view list rather than written down, so these tests
check the derivation rather than a hardcoded table.
"""

from __future__ import annotations

import pytest

from vr_compose.rig import (
    REGISTRY,
    Rig,
    UnknownRigError,
    View,
    fifteen_file_rig,
    rig_for,
    twenty_file_rig,
)


def test_view_normalises_yaw() -> None:
    assert View(360.0, 10.0).normalised() == View(0.0, 10.0)
    assert View(-72.0, 0.0).normalised() == View(288.0, 0.0)
    assert View(432.0, 45.0).normalised() == View(72.0, 45.0)


def test_twenty_file_rig_shape() -> None:
    rig = twenty_file_rig()
    assert rig.file_count == 20
    assert rig.fov_deg == 90.0
    assert rig.mirrored is True
    assert {v.yaw for v in rig.views} == {0.0, 72.0, 144.0, 216.0, 288.0}
    assert {v.elevation for v in rig.views} == {45.0, -45.0, 0.0}


def test_twenty_file_rig_has_fifteen_distinct_viewpoints() -> None:
    rig = twenty_file_rig()
    assert len(rig.groups) == 15
    assert len(rig.unique_indices) == 15
    assert len(rig.duplicate_indices) == 5
    assert set(rig.unique_indices) | set(rig.duplicate_indices) == set(range(1, 21))
    assert not set(rig.unique_indices) & set(rig.duplicate_indices)


def test_twenty_file_rig_duplicate_groups_are_the_sector_boundaries() -> None:
    """Every sector's 4th file re-renders the next sector's 1st; the last wraps to file 1."""
    groups = {group for group in twenty_file_rig().groups if len(group) > 1}
    assert groups == {(1, 20), (4, 5), (8, 9), (12, 13), (16, 17)}


def test_unique_views_cover_every_distinct_orientation() -> None:
    rig = twenty_file_rig()
    unique = rig.unique_views
    assert len(unique) == 15
    assert {v.normalised() for v in unique.values()} == {v.normalised() for v in rig.views}


def test_each_sector_has_three_elevations() -> None:
    rig = twenty_file_rig()
    for yaw in (0.0, 72.0, 144.0, 216.0, 288.0):
        elevations = {v.elevation for v in rig.unique_views.values() if v.yaw == yaw}
        assert elevations == {45.0, -45.0, 0.0}, f"sector {yaw} is incomplete"


@pytest.mark.parametrize("index", [0, -1, 21, 100])
def test_view_for_rejects_out_of_range_indices(index: int) -> None:
    with pytest.raises(ValueError, match="camera index"):
        twenty_file_rig().view_for(index)


def test_view_for_is_one_based() -> None:
    rig = twenty_file_rig()
    assert rig.view_for(1) == rig.views[0]
    assert rig.view_for(20) == rig.views[19]


def test_rig_rejects_degenerate_definitions() -> None:
    with pytest.raises(ValueError, match="at least 2 views"):
        Rig(name="tiny", views=(View(0.0, 0.0),))
    with pytest.raises(ValueError, match="invalid fov"):
        Rig(name="wide", views=(View(0.0, 0.0), View(90.0, 0.0)), fov_deg=180.0)


def test_fifteen_file_rig_is_the_twenty_file_rigs_distinct_views_renumbered() -> None:
    fifteen, twenty = fifteen_file_rig(), twenty_file_rig()
    assert fifteen.file_count == 15
    assert fifteen.views == tuple(twenty.unique_views.values())
    assert (fifteen.fov_deg, fifteen.mirrored) == (twenty.fov_deg, twenty.mirrored)
    assert fifteen.unique_indices == tuple(range(1, 16))
    assert fifteen.duplicate_indices == ()


def test_fifteen_file_rig_matches_the_roadmap_table() -> None:
    """ROADMAP P11: camera k is sector (k-1)//3 at +45, -45, 0 for slot (k-1)%3."""
    rig = fifteen_file_rig()
    for k in range(1, 16):
        sector, slot = divmod(k - 1, 3)
        assert rig.view_for(k) == View(72.0 * sector, (45.0, -45.0, 0.0)[slot]), k


def test_twenty_to_fifteen_numbering() -> None:
    """ROADMAP P11's correspondence: 20-file `i` (not a multiple of 4) is 15-file `k`."""
    fifteen, twenty = fifteen_file_rig(), twenty_file_rig()
    for i in (i for i in range(1, 21) if i % 4):
        k = 3 * ((i - 1) // 4) + (i - 1) % 4 + 1
        assert twenty.view_for(i).normalised() == fifteen.view_for(k).normalised(), (i, k)
    # and the files the 20-file rig actually reads map onto 1..15 in order
    for k, i in enumerate(twenty.unique_indices, start=1):
        assert twenty.view_for(i) == fifteen.view_for(k)


def test_only_the_fifteen_file_layout_needs_a_layout_check() -> None:
    """Contiguous 1..20 proves the 20-file layout; 1..15 could be a truncated 20."""
    assert fifteen_file_rig().needs_layout_check is True
    assert twenty_file_rig().needs_layout_check is False


def test_registry_resolves_the_known_layouts() -> None:
    assert rig_for(20).name == "of3d-20"
    assert rig_for(15).name == "of3d-15"
    assert set(REGISTRY) == {15, 20}


@pytest.mark.parametrize("count", [2, 6, 12, 19, 21, 40])
def test_unknown_layout_is_refused_with_instructions(count: int) -> None:
    """Guessing a rig would produce a plausible-looking, geometrically wrong panorama."""
    with pytest.raises(UnknownRigError) as caught:
        rig_for(count)
    message = str(caught.value)
    assert str(count) in message
    assert "fit_rig" in message, "the error must say how to solve an unknown layout"


def test_native_width_is_the_tile_centre_density() -> None:
    """AGENTS.md section 3: 1920 px / 90 deg == 7680 px / 360 deg, an exact match.

    This is what ties the master size to the render: 1920 tiles give the 8K master,
    and a 16K render (3840 tiles) gives a 15360-wide one.
    """
    rig = twenty_file_rig()
    assert rig.fov_deg == 90.0
    assert rig.native_width(1920) == 7680
    assert rig.native_width(3840) == 15360
    assert rig.native_width(64) == 256
    # a hypothetical narrower rig scales the same way, and stays even
    assert Rig("narrow", rig.views, fov_deg=72.0).native_width(1920) == 9600
    assert Rig("odd", rig.views, fov_deg=90.0).native_width(3) % 2 == 0
    with pytest.raises(ValueError, match="tile size"):
        rig.native_width(0)
