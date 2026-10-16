"""Do the cameras of one frame share one animation state?

ROADMAP P13. A rig that renders one frame from one world state draws every moving object
-- fish, particles, weed -- in the same direction in every camera that sees it. Camera360
did not (AGENTS.md section 3, the Bay render): its simulation advanced or was re-seeded
between cameras, so a fish sat in one place for one camera and elsewhere for its
neighbour, and the blend turned it half transparent. The upstream pass VROpPanoramic
renders all views inside one engine frame (its ROADMAP C4); this probe is the measurement
that says so from the tiles alone.

For each camera pair the overlap is scanned, in A's own warp, for small isolated objects
(`blend_probe.find_objects`). An object is *moving* when it tracks from frame t-1 to t
with a consistent shape and a displacement of at least a pixel -- caustic flicker and
static texture do not track. Each moving object is then searched for in B at the same
direction by normalised cross-correlation, within a radius given in degrees, and three
things are reported per pair and per frame:

1. **found within r**, for r = 0.25 / 0.5 / 1 deg and the search radius: the share of
   moving objects whose best match within r keeps at least half of A's contrast. The last
   point is the "no correspondence" share AGENTS.md section 3 quoted, which turned out to
   be a shape-match failure rate rather than absence; the shape failure rate (NCC below
   0.5) is printed beside it for that comparison.
2. **residual displacement** after the pair's static offset (the layout gate's shift on
   the overlap's static texture, in this sign; the matches' own median when the gate has
   no patches, and it is printed beside the gate's for comparison): median and p90,
   and its decomposition along and across each object's own motion, in frames of that
   motion. A clock or seed difference between cameras shows up here as a scatter.
3. **frame offset**: the same residual with A(t) matched against B(t-1), B(t) and B(t+1).
   One shared state reads about 0 at t and about one frame's motion at t+-1; a camera one
   frame behind reads the opposite.

    python tools/correspondence_probe.py --source E:/VR-DEV/Render/VROpPanoramic/C4_noPP/<run>
    python tools/correspondence_probe.py --source E:/360/0910B --frames 900 903 --pair 3 7
    python tools/correspondence_probe.py --source E:/360/0910B --frames 900 903 --out-dir probe_out

Default pairs are the horizon ring's five neighbours plus the layout gate's ten. Read-only
on the source directory; `--out-dir` receives a crop of every object not found.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
from collections.abc import Iterable, Sequence

import numpy as np
import numpy.typing as npt
from align_probe import warp_one
from blend_probe import MIN_CONTRAST, OBJECT, find_objects, strength
from PIL import Image

from vr_compose import io, layout, source
from vr_compose.rig import Rig, rig_for

F64 = npt.NDArray[np.float64]
Bool = npt.NDArray[np.bool_]
U8 = npt.NDArray[np.uint8]
DEFAULT_ROOT = pathlib.Path("E:/360/0910B")

HALF = 10
"""Half-width of the patch correlated, in output pixels: wider than `blend_probe.OBJECT`,
so an object's outline and not only its centre decides a match."""
SEARCH_DEG = 2.8
"""Default search radius: the 30 px at 3840 wide of AGENTS.md section 3's analysis."""
CURVE_DEG = (0.25, 0.5, 1.0)
"""Radii of the found-within curve besides the search radius itself."""
TRACK_NCC = 0.6
"""An object tracks from t-1 to t only above this correlation."""
SHAPE_NCC = 0.5
"""Below this a match in B is a shape failure; counted beside the absence share."""
MIN_MOVE_PX = 1.0
"""Displacement from t-1 to t below which an object is static, in output pixels."""
KEPT_CONTRAST = 0.5
"""A match is present when B keeps at least this share of A's contrast."""
MIN_STATIC = 5
"""Matches a pair needs before its own median displacement is the static offset."""
MAX_MOVERS = 200
"""Moving objects kept per pair and frame, evenly thinned; the Bay render offers 500."""
OFFSETS = (-1, 0, 1)
"""B frames matched against A(t), relative to t."""


def erode(mask: Bool, radius: int) -> Bool:
    """True where the whole `(2 * radius + 1)` square is true; the frame edge counts as
    false, so no window later reaches outside the array."""
    if radius <= 0:
        return mask.copy()
    padded = np.pad(mask.astype(np.int32), radius + 1)
    table = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    n = 2 * radius + 1
    height, width = mask.shape
    total = (
        table[n : n + height, n : n + width]
        - table[0:height, n : n + width]
        - table[n : n + height, 0:width]
        + table[0:height, 0:width]
    )
    return np.asarray(total == n * n, dtype=np.bool_)


def correlate(patch: F64, target: F64, y: int, x: int, radius: int) -> F64:
    """Normalised cross-correlation of `patch` at every position within `radius` of
    `(y, x)` in `target`, as a `(2 * radius + 1)` square map centred on zero shift."""
    half = patch.shape[0] // 2
    window = np.asarray(
        target[
            y - radius - half : y + radius + half + 1, x - radius - half : x + radius + half + 1
        ],
        dtype=np.float64,
    )
    views = np.lib.stride_tricks.sliding_window_view(window, patch.shape)
    flat = views.reshape(views.shape[0], views.shape[1], -1)
    centred = flat - flat.mean(axis=2, keepdims=True)
    norms = np.sqrt((centred * centred).sum(axis=2))
    p = (patch - patch.mean()).ravel()
    ncc = (centred @ p) / np.maximum(norms * np.sqrt((p * p).sum()), 1e-9)
    return np.asarray(ncc, dtype=np.float64)


def _parabola(left: float, centre: float, right: float) -> float:
    """Sub-pixel vertex of a peak sampled at -1, 0, +1; 0 when the three do not form one."""
    denominator = left - 2.0 * centre + right
    if denominator >= 0.0:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))


def best_within(ncc: F64, radius: int, limit_px: float) -> tuple[float, float, float] | None:
    """`(dy, dx, ncc)` of the best position within `limit_px` of zero shift, refined to
    sub-pixel; None when no position lies within the limit."""
    offsets = np.arange(ncc.shape[0], dtype=np.float64) - radius
    distance = np.hypot(offsets[:, None], offsets[None, :])
    masked = np.where(distance <= limit_px, ncc, -np.inf)
    iy, ix = (int(v) for v in np.unravel_index(int(np.argmax(masked)), masked.shape))
    if not np.isfinite(masked[iy, ix]):
        return None
    dy, dx = float(offsets[iy]), float(offsets[ix])
    last = ncc.shape[0] - 1
    if 0 < iy < last:
        dy += _parabola(float(ncc[iy - 1, ix]), float(ncc[iy, ix]), float(ncc[iy + 1, ix]))
    if 0 < ix < last:
        dx += _parabola(float(ncc[iy, ix - 1]), float(ncc[iy, ix]), float(ncc[iy, ix + 1]))
    return dy, dx, float(ncc[iy, ix])


@dataclasses.dataclass(frozen=True, slots=True)
class Mover:
    """A moving object in A's warp of frame t: where it is, how strong, and its motion
    from t-1 in output pixels per frame."""

    y: int
    x: int
    contrast: float
    sign: float
    vy: float
    vx: float

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vy, self.vx))


def moving_objects(previous: F64, now: F64, usable: Bool, radius: int) -> list[Mover]:
    """Isolated objects of `now` that track back into `previous` and moved."""
    movers: list[Mover] = []
    for y, x in find_objects(now, usable):
        contrast, sign = strength(now, y, x)
        if contrast < MIN_CONTRAST:
            continue
        patch = now[y - HALF : y + HALF + 1, x - HALF : x + HALF + 1]
        found = best_within(correlate(patch, previous, y, x, radius), radius, radius)
        if found is None:
            continue
        dy, dx, ncc = found
        if ncc < TRACK_NCC or np.hypot(dy, dx) < MIN_MOVE_PX:
            continue
        movers.append(Mover(y, x, contrast, sign, -dy, -dx))
    return movers


def kept_share(mover: Mover, target: F64, dy: float, dx: float) -> float:
    """B's contrast at the matched position as a share of A's, with A's sign imposed."""
    y, x = mover.y + round(dy), mover.x + round(dx)
    return strength(target, y, x, mover.sign)[0] / mover.contrast


@dataclasses.dataclass(slots=True)
class PairFrame:
    """One pair's numbers on one frame."""

    a: int
    b: int
    frame: int
    movers: int = 0
    found: dict[float, int] = dataclasses.field(default_factory=dict)
    shape_failures: int = 0
    static: tuple[float, float] | None = None
    own: tuple[float, float] | None = None
    gate: tuple[float, float] | None = None
    residual_px: list[float] = dataclasses.field(default_factory=list)
    along_frames: list[float] = dataclasses.field(default_factory=list)
    across_frames: list[float] = dataclasses.field(default_factory=list)
    offsets_px: dict[int, list[float]] = dataclasses.field(default_factory=dict)
    misses: list[Mover] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True, slots=True)
class Totals:
    """`PairFrame` numbers pooled over frames, or over everything."""

    movers: int
    found: dict[float, int]
    shape_failures: int
    residual_px: list[float]
    along_frames: list[float]
    across_frames: list[float]
    offsets_px: dict[int, list[float]]

    @classmethod
    def of(cls, results: Iterable[PairFrame]) -> Totals:
        pooled = Totals(0, {}, 0, [], [], [], {k: [] for k in OFFSETS})
        movers, failures = 0, 0
        for result in results:
            movers += result.movers
            failures += result.shape_failures
            for radius, count in result.found.items():
                pooled.found[radius] = pooled.found.get(radius, 0) + count
            pooled.residual_px.extend(result.residual_px)
            pooled.along_frames.extend(result.along_frames)
            pooled.across_frames.extend(result.across_frames)
            for k, values in result.offsets_px.items():
                pooled.offsets_px[k].extend(values)
        return dataclasses.replace(pooled, movers=movers, shape_failures=failures)

    def found_share(self, radius: float) -> float:
        return self.found.get(radius, 0) / self.movers if self.movers else 0.0


def ring_pairs(rig: Rig) -> list[tuple[int, int]]:
    """Each horizon camera against the next one round the ring."""
    horizon = sorted(
        (i for i in rig.unique_indices if rig.view_for(i).elevation == 0.0),
        key=lambda i: rig.view_for(i).yaw % 360.0,
    )
    return [(horizon[k], horizon[(k + 1) % len(horizon)]) for k in range(len(horizon))]


def default_pairs(rig: Rig) -> list[tuple[int, int]]:
    return ring_pairs(rig) + layout.pairs_for(rig, "all")


class Warps:
    """Each camera's luma on the equirect grid, per frame, kept for the frames in play."""

    def __init__(self, chosen: source.SourceSet, rig: Rig, width: int) -> None:
        self.chosen, self.rig, self.width = chosen, rig, width
        self.size = chosen.tile_size or 0
        self.tiles: dict[int, dict[int, U8]] = {}
        self.cache: dict[tuple[int, int], tuple[F64, Bool]] = {}

    def load(self, frame: int, cameras: Sequence[int]) -> dict[int, U8]:
        if frame not in self.tiles:
            loaded = io.load_tiles(self.chosen, frame, list(cameras), workers=8)
            self.tiles[frame] = io.uint8_tiles(loaded)
        return self.tiles[frame]

    def get(self, camera: int, frame: int, cameras: Sequence[int]) -> tuple[F64, Bool]:
        key = (camera, frame)
        if key not in self.cache:
            tiles = self.load(frame, cameras)
            self.cache[key] = warp_one(tiles, self.rig, camera, self.size, self.width)
        return self.cache[key]

    def forget_before(self, frame: int) -> None:
        for key in [k for k in self.cache if k[1] < frame]:
            del self.cache[key]
        for old in [f for f in self.tiles if f < frame]:
            del self.tiles[old]


def analyse_pair(
    a_prev: F64,
    a_now: F64,
    seen_a: Bool,
    b_by_offset: dict[int, F64],
    seen_b: Bool,
    *,
    radius: int,
    search_deg: float,
    degrees: float,
    a: int,
    b: int,
    frame: int,
    gate: tuple[float, float] | None = None,
) -> PairFrame:
    """The three numbers for one pair on one frame. `b_by_offset` holds B's warps keyed
    by frame offset from t; 0 is required, -1 and +1 are the frame-offset check. `radius`
    is `search_deg` in pixels, and keys the found curve's last point by the degrees asked."""
    result = PairFrame(a, b, frame, gate=gate)
    usable = erode(seen_a & seen_b, radius + HALF + 2 * OBJECT + 1)
    movers = moving_objects(a_prev, a_now, usable, radius)
    if len(movers) > MAX_MOVERS:
        movers = movers[:: -(-len(movers) // MAX_MOVERS)]
    result.movers = len(movers)
    radii = [*CURVE_DEG, search_deg]
    result.found = {r: 0 for r in radii}
    result.offsets_px = {k: [] for k in OFFSETS if k in b_by_offset}
    b_now = b_by_offset[0]

    matched: list[tuple[Mover, float, float]] = []
    maps: list[tuple[Mover, F64]] = []
    for mover in movers:
        patch = a_now[mover.y - HALF : mover.y + HALF + 1, mover.x - HALF : mover.x + HALF + 1]
        ncc = correlate(patch, b_now, mover.y, mover.x, radius)
        maps.append((mover, ncc))
        for r in radii:
            best = best_within(ncc, radius, radius if r == search_deg else r / degrees)
            if best is not None and kept_share(mover, b_now, best[0], best[1]) >= KEPT_CONTRAST:
                result.found[r] += 1
        best = best_within(ncc, radius, radius)
        if best is None or best[2] < SHAPE_NCC:
            result.shape_failures += 1
            continue
        if kept_share(mover, b_now, best[0], best[1]) < KEPT_CONTRAST:
            result.misses.append(mover)
            continue
        matched.append((mover, best[0], best[1]))

    if len(matched) >= MIN_STATIC:
        shifts = np.array([(dy, dx) for _, dy, dx in matched])
        result.own = (float(np.median(shifts[:, 0])), float(np.median(shifts[:, 1])))
    # The gate measures the overlap's static texture, which is what "static" means; a
    # pair's own median is a coin toss when its movers are few or mostly texture.
    static = gate or result.own or (0.0, 0.0)
    result.static = static
    for mover, dy, dx in matched:
        ry, rx = dy - static[0], dx - static[1]
        result.residual_px.append(float(np.hypot(ry, rx)))
        speed = mover.speed
        uy, ux = mover.vy / speed, mover.vx / speed
        result.along_frames.append((ry * uy + rx * ux) / speed)
        result.across_frames.append((rx * uy - ry * ux) / speed)

    for k, target in b_by_offset.items():
        if k not in result.offsets_px:
            continue
        for mover, ncc in maps:
            if k != 0:
                patch = a_now[
                    mover.y - HALF : mover.y + HALF + 1, mover.x - HALF : mover.x + HALF + 1
                ]
                ncc = correlate(patch, target, mover.y, mover.x, radius)
            best = best_within(ncc, radius, radius)
            if best is None or best[2] < SHAPE_NCC:
                continue
            if kept_share(mover, target, best[0], best[1]) < KEPT_CONTRAST:
                continue
            result.offsets_px[k].append(float(np.hypot(best[0] - static[0], best[1] - static[1])))
    return result


def probe_frame(
    warps: Warps,
    frame: int,
    pairs: Sequence[tuple[int, int]],
    *,
    radius: int,
    search_deg: float,
    with_gate: bool = True,
) -> list[PairFrame]:
    """Every pair's numbers on `frame`, which needs frame - 1 and uses frame + 1 if present."""
    cameras = sorted({c for pair in pairs for c in pair})
    frames = warps.chosen.frames
    degrees = 360.0 / warps.width
    results = []
    for a, b in pairs:
        a_prev, _ = warps.get(a, frame - 1, cameras)
        a_now, seen_a = warps.get(a, frame, cameras)
        b_by_offset: dict[int, F64] = {}
        seen_b: Bool | None = None
        for k in OFFSETS:
            if frame + k in frames:
                luma, seen = warps.get(b, frame + k, cameras)
                b_by_offset[k] = luma
                if k == 0:
                    seen_b = seen
        assert seen_b is not None
        gate = None
        if with_gate:
            shift = layout.pair_shifts(warps.load(frame, cameras), warps.rig, a, b, warps.width)
            if shift.checked:
                # the gate's shift carries B to A; this probe's displacement is where A's
                # object sits in B, so the sign flips
                gate = (
                    -float(np.mean([s[1] for s in shift.shifts])),
                    -float(np.mean([s[2] for s in shift.shifts])),
                )
        results.append(
            analyse_pair(
                a_prev,
                a_now,
                seen_a,
                b_by_offset,
                seen_b,
                radius=radius,
                search_deg=search_deg,
                degrees=degrees,
                a=a,
                b=b,
                frame=frame,
                gate=gate,
            )
        )
    return results


def _median(values: Sequence[float]) -> float:
    return float(np.median(values)) if values else float("nan")


def _p90(values: Sequence[float]) -> float:
    return float(np.percentile(values, 90)) if values else float("nan")


def totals_line(label: str, totals: Totals, degrees: float, radii: Sequence[float]) -> str:
    found = "  ".join(f"{r:g} {totals.found_share(r) * 100:3.0f}%" for r in radii)
    failures = totals.shape_failures / totals.movers * 100 if totals.movers else 0.0
    residual = _median(totals.residual_px)
    offsets = "  ".join(
        f"t{k:+d} {_median(v):4.1f}" if k else f"t {_median(v):4.1f}"
        for k, v in sorted(totals.offsets_px.items())
    )
    p90 = _p90(totals.residual_px)
    along, across = _median(totals.along_frames), _median(totals.across_frames)
    return (
        f"{label}: movers {totals.movers:4d}  found(deg) {found}"
        f"  shape<{SHAPE_NCC:g} {failures:3.0f}%"
        f"  residual med {residual:4.2f} px ({residual * degrees:.3f} deg) p90 {p90:4.2f}"
        f"  along {along:+.2f} across {across:+.2f} frames  offset px {offsets}"
    )


def pair_line(result: PairFrame, degrees: float, radii: Sequence[float]) -> str:
    label = f"f{result.frame:04d} C{result.a:<2d}/C{result.b:<2d}"
    line = totals_line(label, Totals.of([result]), degrees, radii)
    if result.static is not None:
        own = f" (own {result.own[0]:+.1f} {result.own[1]:+.1f})" if result.own else ""
        source = "gate" if result.gate else "own" if result.own else "none"
        line += f"  static {result.static[0]:+.1f} {result.static[1]:+.1f} {source}{own}"
    return line


def write_crops(
    out_dir: pathlib.Path, result: PairFrame, a_now: F64, b_now: F64, radius: int
) -> None:
    """A's object beside B's search window, for every object not found."""
    out_dir.mkdir(parents=True, exist_ok=True)
    margin = radius + HALF
    for i, mover in enumerate(result.misses):
        rows = slice(mover.y - margin, mover.y + margin + 1)
        cols = slice(mover.x - margin, mover.x + margin + 1)
        side_by_side = np.concatenate([a_now[rows, cols], b_now[rows, cols]], axis=1)
        image = np.clip(side_by_side, 0, 255).astype(np.uint8)
        name = f"f{result.frame:04d}_C{result.a}_C{result.b}_{i:03d}.png"
        Image.fromarray(image).save(out_dir / name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=pathlib.Path, default=DEFAULT_ROOT)
    parser.add_argument("--stem", default=None)
    parser.add_argument(
        "--frames", type=int, nargs=2, default=None, metavar=("FIRST", "LAST"),
        help="frames to analyse, inclusive; each needs its predecessor (default: five)",
    )  # fmt: skip
    parser.add_argument(
        "--width", type=int, default=None, help="default: the layout gate's analysis width"
    )
    parser.add_argument(
        "--pair", type=int, nargs=2, default=None, metavar=("A", "B"),
        help="one pair instead of the default fifteen",
    )  # fmt: skip
    parser.add_argument("--radius-deg", type=float, default=SEARCH_DEG)
    parser.add_argument("--no-gate", action="store_true", help="skip the layout gate's shift")
    parser.add_argument("--out-dir", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)

    sets, searched = source.discover(args.source)
    if args.stem:
        sets = [s for s in sets if s.stem == args.stem]
    if len(sets) != 1:
        raise SystemExit(f"need exactly one source set (got {[s.stem for s in sets]}; {searched})")
    chosen = sets[0]
    rig = rig_for(chosen.camera_count)
    size = chosen.tile_size
    if size is None:
        raise SystemExit("source tiles are not square or not uniform")
    width = args.width or layout.analysis_width(rig, size)
    degrees = 360.0 / width
    radius = round(args.radius_deg / degrees)
    frames = list(chosen.frames)
    if args.frames is None:
        wanted = frames[1:6]
    else:
        wanted = [f for f in frames if args.frames[0] <= f <= args.frames[1]]
    wanted = [f for f in wanted if f - 1 in chosen.frames]
    if not wanted:
        raise SystemExit("no frame to analyse has its predecessor in the source")
    pairs = [tuple(args.pair)] if args.pair else default_pairs(rig)
    radii = [*CURVE_DEG, args.radius_deg]

    print(f"source : {chosen.root} stem {chosen.stem!r}, tile {size}")
    print(
        f"scan   : {width} wide, 1 px = {degrees:.3f} deg, search {args.radius_deg:g} deg = "
        f"{radius} px, patch {2 * HALF + 1} px, {len(pairs)} pair(s), "
        f"frames {wanted[0]}..{wanted[-1]}\n"
    )
    warps = Warps(chosen, rig, width)
    per_pair: dict[tuple[int, int], list[PairFrame]] = {p: [] for p in pairs}
    for frame in wanted:
        results = probe_frame(
            warps, frame, pairs, radius=radius, search_deg=args.radius_deg,
            with_gate=not args.no_gate,
        )  # fmt: skip
        for result in results:
            per_pair[(result.a, result.b)].append(result)
            print(pair_line(result, degrees, radii))
            if args.out_dir and result.misses:
                cameras = sorted({c for pair in pairs for c in pair})
                a_now, _ = warps.get(result.a, frame, cameras)
                b_now, _ = warps.get(result.b, frame, cameras)
                write_crops(args.out_dir, result, a_now, b_now, radius)
        warps.forget_before(frame)

    print()
    for (a, b), results in per_pair.items():
        print(totals_line(f"C{a:<2d}/C{b:<2d} all", Totals.of(results), degrees, radii))
    everything = Totals.of(r for results in per_pair.values() for r in results)
    print("\n" + totals_line("overall", everything, degrees, radii))
    print(
        "\nOne shared animation state: found 100% at the search radius, residual at the "
        "noise floor, along and across about 0 frames, offset about 0 at t and about one "
        "frame's motion at t+-1."
    )
    return 0 if everything.movers else 1


if __name__ == "__main__":
    raise SystemExit(main())
