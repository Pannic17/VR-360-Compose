"""Rig definitions and the registry.

The 20-file layout's most useful property -- that 20 files hold only 15 distinct
viewpoints -- is *derived* from the view list rather than written down, so these tests
check the derivation rather than a hardcoded table.
"""

from __future__ import annotations

import pytest

from vr_compose.rig import REGISTRY, Rig, UnknownRigError, View, rig_for, twenty_file_rig


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


def test_registry_resolves_the_known_layout() -> None:
    assert rig_for(20).name == "of3d-20"
    assert set(REGISTRY) == {20}


@pytest.mark.parametrize("count", [2, 6, 12, 19, 21, 40])
def test_unknown_layout_is_refused_with_instructions(count: int) -> None:
    """Guessing a rig would produce a plausible-looking, geometrically wrong panorama."""
    with pytest.raises(UnknownRigError) as caught:
        rig_for(count)
    message = str(caught.value)
    assert str(count) in message
    assert "fit_rig" in message, "the error must say how to solve an unknown layout"
