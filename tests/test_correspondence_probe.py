"""The correspondence probe (ROADMAP P13): does it read back what was put in?

Synthetic frames: a flat panorama with bright dots that each move at their own velocity,
rendered into 15 tiles at 512 px for frames 0..3. Every camera sampling the same panorama
is one shared animation state, which must read 100% found, a residual at the noise floor
and the frame-offset signature. One camera's tiles then come from a panorama where a tenth
of the dots are missing and the rest sit 0.3 deg off; the probe must read those two numbers
back from that camera's pairs, and nothing from the others. That is the calibration the
hand analysis of AGENTS.md section 3 never had.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import numpy.typing as npt
import pytest

from conftest import make_source_tree, sample_panorama_into_tile
from vr_compose import source
from vr_compose.rig import fifteen_file_rig

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))

import correspondence_probe as probe

U8 = npt.NDArray[np.uint8]
TILE = 512
WIDTH = 2048
HEIGHT = WIDTH // 2
DEGREES = 360.0 / WIDTH
FRAMES = (0, 1, 2, 3)
INJECTED = 3
"""The horizon camera of sector 0, whose tiles carry the injected defects."""
MISSING = 0.10
SCATTER_DEG = 0.3
BACKGROUND = 90
DOT = 230


def _dots() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """`(positions (n, 2) as (v, u), velocities (n, 2))` on a jittered grid between
    +-60 deg latitude, 48 px apart so each dot sits alone in its ring."""
    rng = np.random.default_rng(7)
    rows = np.arange(int(HEIGHT * (0.5 - 60.0 / 180.0)), int(HEIGHT * (0.5 + 60.0 / 180.0)), 48)
    cols = np.arange(40, WIDTH - 40, 48)
    v, u = np.meshgrid(rows, cols, indexing="ij")
    positions = np.stack([v.ravel(), u.ravel()], axis=1).astype(np.float64)
    positions += rng.uniform(-10.0, 10.0, positions.shape)
    speed = rng.uniform(3.0, 7.0, len(positions))
    angle = rng.uniform(0.0, 2.0 * np.pi, len(positions))
    velocities = np.stack([speed * np.sin(angle), speed * np.cos(angle)], axis=1)
    return positions, velocities


POSITIONS, VELOCITIES = _dots()
_rng = np.random.default_rng(11)
REMOVED = _rng.random(len(POSITIONS)) < MISSING
OFFSETS = _rng.normal(0.0, SCATTER_DEG / DEGREES, POSITIONS.shape)


def panorama(frame: int, *, injected: bool) -> U8:
    image = np.full((HEIGHT, WIDTH, 3), BACKGROUND, np.uint8)
    for i, (position, velocity) in enumerate(zip(POSITIONS, VELOCITIES, strict=True)):
        if injected and REMOVED[i]:
            continue
        v, u = position + frame * velocity + (OFFSETS[i] if injected else 0.0)
        iv, iu = round(v), round(u) % WIDTH
        if not 2 <= iv < HEIGHT - 2:
            continue
        image[iv - 2 : iv + 3, max(0, iu - 2) : iu + 3] = DOT
    return image


def _tree(root: pathlib.Path, *, injected_camera: int | None) -> source.SourceSet:
    rig = fifteen_file_rig()
    panoramas = {f: panorama(f, injected=False) for f in FRAMES}
    altered = {f: panorama(f, injected=True) for f in FRAMES} if injected_camera else {}

    def tile_for(index: int, frame: int) -> U8:
        image = altered[frame] if index == injected_camera else panoramas[frame]
        return sample_panorama_into_tile(image, rig, index, TILE)

    make_source_tree(
        root, cameras=15, stem="Dots", frames=FRAMES, inner=None, size=(TILE, TILE),
        tile_for=tile_for, name_digits=2,
    )  # fmt: skip
    return source.scan(root)[0]


@pytest.fixture(scope="module")
def shared(tmp_path_factory: pytest.TempPathFactory) -> source.SourceSet:
    return _tree(tmp_path_factory.mktemp("shared"), injected_camera=None)


@pytest.fixture(scope="module")
def injected(tmp_path_factory: pytest.TempPathFactory) -> source.SourceSet:
    return _tree(tmp_path_factory.mktemp("injected"), injected_camera=INJECTED)


def run(chosen: source.SourceSet, pairs: list[tuple[int, int]]) -> probe.Totals:
    warps = probe.Warps(chosen, fifteen_file_rig(), WIDTH)
    radius = round(probe.SEARCH_DEG / DEGREES)
    results = [
        r
        for frame in (1, 2)
        for r in probe.probe_frame(
            warps, frame, pairs, radius=radius, search_deg=probe.SEARCH_DEG, with_gate=False
        )
    ]
    return probe.Totals.of(results)


def test_the_default_pairs_are_the_ring_and_the_gate() -> None:
    rig = fifteen_file_rig()
    assert probe.ring_pairs(rig) == [(3, 6), (6, 9), (9, 12), (12, 15), (15, 3)]
    assert len(probe.default_pairs(rig)) == 15


def test_erode_treats_the_edge_as_false() -> None:
    mask = np.ones((9, 9), bool)
    mask[4, 4] = False
    eroded = probe.erode(mask, 1)
    assert not eroded[0].any() and not eroded[:, 0].any()
    assert not eroded[3:6, 3:6].any()
    assert eroded[1, 1] and eroded[7, 7]
    assert probe.erode(mask, 0).tolist() == mask.tolist()


def test_best_within_refines_to_sub_pixel() -> None:
    radius = 5
    offsets = np.arange(2 * radius + 1) - radius
    peak_y, peak_x = 1.3, -2.4
    ncc = np.exp(-((offsets[:, None] - peak_y) ** 2 + (offsets[None, :] - peak_x) ** 2) / 4.0)
    found = probe.best_within(ncc, radius, radius)
    assert found is not None
    assert abs(found[0] - peak_y) < 0.15 and abs(found[1] - peak_x) < 0.15
    assert probe.best_within(ncc, radius, 0.5) is not None  # the centre is always within
    assert probe.best_within(ncc, radius, -1.0) is None


@pytest.mark.slow
def test_one_shared_state_reads_everything_found(shared: source.SourceSet) -> None:
    totals = run(shared, probe.default_pairs(fifteen_file_rig()))
    search = probe.SEARCH_DEG
    assert totals.movers > 200
    assert totals.found_share(search) == 1.0
    assert totals.found_share(0.5) > 0.95
    assert totals.shape_failures == 0
    assert np.median(totals.residual_px) < 0.7
    assert abs(np.median(totals.along_frames)) < 0.15
    assert abs(np.median(totals.across_frames)) < 0.15
    # the frame-offset signature: nothing at t, one frame's motion either side
    assert np.median(totals.offsets_px[0]) < 0.7
    for k in (-1, 1):
        assert 2.0 < np.median(totals.offsets_px[k]) < 9.0, k


@pytest.mark.slow
def test_an_injected_camera_reads_its_absence_and_scatter(injected: source.SourceSet) -> None:
    rig = fifteen_file_rig()
    as_b = [(a, b) for a, b in probe.default_pairs(rig) if b == INJECTED]
    assert as_b == [(15, 3), (2, 3), (1, 3)]
    defective = run(injected, as_b)
    assert defective.movers > 60
    # a tenth of the dots are gone: not found at any radius, found at none
    assert 0.80 < defective.found_share(probe.SEARCH_DEG) < 0.95
    # the rest sit 0.3 deg off: 1.7 px per axis at this width, so about 2 px of residual
    assert np.median(defective.residual_px) > 1.2
    assert np.median(defective.residual_px) * DEGREES < 0.6
    # a pair that does not touch the injected camera reads clean
    clean = run(injected, [(5, 6)])
    assert clean.found_share(probe.SEARCH_DEG) == 1.0
    assert np.median(clean.residual_px) < 0.7


@pytest.mark.slow
def test_the_command_line_runs_one_pair(
    shared: source.SourceSet, capsys: pytest.CaptureFixture[str]
) -> None:
    code = probe.main(
        ["--source", str(shared.root), "--frames", "1", "2", "--pair", "2", "3", "--no-gate"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "f0001 C2 /C3" in out and "f0002 C2 /C3" in out
    assert "overall: movers" in out
