"""Celtic knot puzzle: read the rings, solve the rotations.

Port of the clue solver's knot reader adapted for scaled captures. The
interface is normalized to 1x from the close button, tiles live on a 24px
isometric lattice discovered by flood fill from a scanned origin, track
membership comes from chromaticity classification (blur-tolerant), and
runes are identified by clustering tile crops against each other, so no
reference rune art is needed. Crossings hidden under the other ring are
constrained by the match/mismatch border and can be filled in by merging
a second read after the user inverts paths or rotates a ring.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from runekit import detection
from .assets import ClueAssets

logger = logging.getLogger(__name__)

KNOT_W, KNOT_H = 504, 326
EOCX_OFFSET = (483, 8)  # close button position within the knot rect
STEP = 24
MAX_TILES = 90  # real knots reach ~50 cells; blur can add phantom neighbours
PAD = 8

# ribbon sample points around the rune glyph, relative to the tile origin
OWN_POINTS = ((23, 13), (13, 23), (33, 23), (23, 33),
              (16, 16), (30, 16), (16, 30), (30, 30))

# normalized chromaticity (255*c/sum) references: -1 background, 0..3 rings
TRACK_CHROMA = {
    -1: (101, 96, 56),
    0: (75, 66, 116),
    1: (144, 107, 6),
    2: (79, 74, 102),
    3: (92, 89, 75),
}
RING_NAMES = {0: "blue", 1: "gold", 2: "black", 3: "silver"}

RUNE_SAME = 0.82  # binarized-glyph agreement above which two runes match
BORDER_MIN = 0.6  # border sprite ZNCC to accept a crossing

# ribbon arms approach the top/bottom corners left and right of the split;
# each exit is voted over points along its arm, margin-gated so blurred
# blends (which land between the color references) don't vote
EXIT_ARMS = (
    ((21, 2), (19, 6), (18, 10)),
    ((26, 2), (28, 6), (29, 10)),
    ((26, 45), (28, 41), (29, 37)),
    ((21, 45), (19, 41), (18, 37)),
)
COLOR_MARGIN = 18

# lattice steps: dir 0..3 = (t,n+1), (t+1,n), (t,n-1), (t-1,n)
DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))


UNKNOWN = -10000  # "not rune 9999": matches nothing, constrains nothing

@dataclass
class KnotSlot:
    x: int
    y: int
    rune: int  # >=0 known rune id; -1-k means "not rune k"; UNKNOWN unreadable


@dataclass
class KnotState:
    paths: List[List[KnotSlot]]
    intersections: List[dict]  # {x, y, col1, i1, col2, i2}
    rune_count: int
    scale: float
    region: Tuple[int, int]  # frame coords of the knot rect


@dataclass
class KnotSolution:
    state: KnotState
    offsets: List[int]
    sure: bool
    candidates: int


class _Reader:
    def __init__(self, norm: np.ndarray, assets: ClueAssets):
        self.norm = norm
        self.border_match = detection.to_array(assets.needle("bordermatch"))
        self.border_nomatch = detection.to_array(assets.needle("bordernomatch"))
        runearea = detection.to_array(assets.needle("runearea"))
        self.rune_mask = runearea[:, :, 3]
        self.rune_size = runearea.shape[:2]
        self.pathor = None
        self.tiles = {}
        self.rune_crops: List[np.ndarray] = []

    def _px(self, x: int, y: int) -> np.ndarray:
        h, w = self.norm.shape[:2]
        x0, x1 = max(0, x - 1), min(w, x + 2)
        y0, y1 = max(0, y - 1), min(h, y + 2)
        if x0 >= x1 or y0 >= y1:
            return np.zeros(3)
        return self.norm[y0:y1, x0:x1, :3].reshape(-1, 3).mean(axis=0)

    def _is_dark(self, x: int, y: int) -> bool:
        r, g, b = self._px(x, y)
        return r < 60 and g < 60 and b < 40

    def track_color(self, x: int, y: int, with_margin: bool = False):
        rgb = self._px(x, y)
        # tracks are dark metallics on a bright background; bright pixels and
        # track/background blends are never ring, whatever their chroma
        if rgb.sum() >= 230:
            return (-1, 999.0) if with_margin else -1
        i = rgb.sum() / 255.0
        if i <= 0:
            return (2, 999.0) if with_margin else 2
        chroma = rgb / i
        best, best_d, second = -1, 1e9, 1e9
        for key, ref in TRACK_CHROMA.items():
            d = float(np.abs(chroma - np.array(ref)).sum())
            if d < best_d:
                best, best_d, second = key, d, best_d
            elif d < second:
                second = d
        if best == 0 and rgb.sum() < 30:
            best = 2
        return (best, second - best_d) if with_margin else best

    def _vote(self, points, sx: int, sy: int, strict: bool = True) -> int:
        votes = []
        for ox, oy in points:
            label, margin = self.track_color(sx + ox, sy + oy, with_margin=True)
            if margin >= COLOR_MARGIN:
                votes.append(label)
        ring_votes = [v for v in votes if v != -1]
        if not ring_votes:
            return -1
        if strict and len(ring_votes) * 2 < len(votes):
            return -1
        return max(set(ring_votes), key=ring_votes.count)

    def origin_candidates(self, count: int = 4) -> List[Tuple[int, int]]:
        """Anchor the lattice on a crossing: the match/nomatch border rings
        are distinctive sprites present in every knot, and a border sits at
        tile origin + (6, 6)."""
        f32 = self.norm[:, :, :3].astype(np.float32)
        found = []
        for spr in (self.border_match, self.border_nomatch):
            res = np.nan_to_num(detection._masked_sqdiff(f32, spr), nan=np.inf)
            for _ in range(3):
                y, x = np.unravel_index(int(res.argmin()), res.shape)
                if not np.isfinite(res[y, x]):
                    break
                res[max(0, y - 12):y + 13, max(0, x - 12):x + 13] = np.inf
                score = detection.zncc(self.norm, spr, x, y)
                if score >= 0.5:
                    found.append((score, x - 6, y - 6))
        found.sort(reverse=True)
        out = []
        for _, x, y in found:
            if all(abs(x - o[0]) + abs(y - o[1]) > 6 for o in out):
                out.append((x, y))
            if len(out) >= count:
                break
        return out

    def _refine_origin(self) -> bool:
        """Snap the scanned point to the nearest lattice origin: any point is
        within half a lattice cell of one, so search that radius for the
        offset where the seed tile classifies as track with connected exits."""
        px, py = self.pathor
        best = None
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                self.pathor = (px + dx, py + dy)
                sx, sy = self._screen(0, 0)
                if self._vote(OWN_POINTS, sx, sy) == -1:
                    continue
                exits = sum(1 for arm in EXIT_ARMS if self._vote(arm, sx, sy) != -1)
                score = exits - (abs(dx) + abs(dy)) * 0.05
                if exits >= 2 and (best is None or score > best[0]):
                    best = (score, px + dx, py + dy)
        if best is None:
            self.pathor = (px, py)
            return False
        self.pathor = (best[1], best[2])
        return True

    def _screen(self, t: int, n: int) -> Tuple[int, int]:
        return (self.pathor[0] + STEP * t - STEP * n, self.pathor[1] - STEP * t - STEP * n)

    def _border(self, sx: int, sy: int) -> Tuple[bool, bool]:
        best_m = best_n = -1.0
        for jx in (-2, -1, 0, 1, 2):
            for jy in (-2, -1, 0, 1, 2):
                best_m = max(best_m, detection.zncc(self.norm, self.border_match, sx + 6 + jx, sy + 6 + jy))
                best_n = max(best_n, detection.zncc(self.norm, self.border_nomatch, sx + 6 + jx, sy + 6 + jy))
        if max(best_m, best_n) < BORDER_MIN:
            return False, False
        return True, best_m >= best_n

    def _rune(self, sx: int, sy: int) -> int:
        """Cluster by binarized glyph shape: bright glyph on dark ribbon, so
        the ink mask is what identifies a rune, not the ribbon tint."""
        h, w = self.rune_size
        J = 3
        crop = self.norm[sy + 11 - J:sy + 11 + h + J, sx + 11 - J:sx + 11 + w + J, :3]
        if crop.shape[:2] != (h + 2 * J, w + 2 * J):
            return -1
        lum = crop.astype(np.float64).sum(axis=2)
        mask = self.rune_mask > 128

        def binarize(a):
            vals = a[J:J + h, J:J + w][mask]
            thr = (vals.min() + vals.max()) / 2
            return a > thr

        binary = binarize(lum)
        windows = np.lib.stride_tricks.sliding_window_view(binary, (h, w))
        flat = windows[..., mask]  # (2J+1, 2J+1, mask px)
        for idx, known in enumerate(self.rune_crops):
            agree = (flat == known[mask]).mean(axis=-1).max()
            if agree >= RUNE_SAME:
                return idx
        if len(self.rune_crops) >= 30:
            return -1
        self.rune_crops.append(binary[J:J + h, J:J + w])
        return len(self.rune_crops) - 1

    def map_tiles(self) -> Optional[str]:
        stack = [(0, 0)]
        while stack:
            t, n = stack.pop()
            if (t, n) in self.tiles:
                continue
            if len(self.tiles) > MAX_TILES:
                return "Too many tiles found; the rings could not be mapped."
            sx, sy = self._screen(t, n)
            exits = [self._vote(arm, sx, sy) for arm in EXIT_ARMS]
            # own ring: plurality over ribbon points around the glyph; the
            # glyph itself dilutes the vote, so no majority requirement
            own = self._vote(OWN_POINTS, sx, sy, strict=False)
            if own == -1 and all(e == -1 for e in exits):
                # background cell; skip the expensive border and rune work
                self.tiles[(t, n)] = {
                    "exits": exits, "own": own, "crossing": False,
                    "matched": False, "rune": -1,
                }
                continue
            crossing, matched = self._border(sx, sy)
            # runes only matter on path tiles; phantom edge cells skip it
            rune = self._rune(sx, sy) if (own != -1 or crossing) else -1
            self.tiles[(t, n)] = {
                "exits": exits, "own": own, "crossing": crossing,
                "matched": matched, "rune": rune,
            }
            for d, (dt, dn) in enumerate(DIRS):
                if exits[d] != -1:
                    stack.append((t + dt, n + dn))
        return None

    def walk_paths(self) -> Tuple[List[List[dict]], List[dict]]:
        paths = [[] for _ in range(4)]
        intersections = []
        for ring in range(4):
            start = None
            best_key = None
            for (t, n), tile in self.tiles.items():
                if tile["own"] == ring:
                    key = 100 * t + n
                    if best_key is None or key > best_key:
                        best_key, start = key, (t, n)
            if start is None:
                continue
            t, n = start
            came_from = -1
            while True:
                if len(paths[ring]) > MAX_TILES:
                    # oscillating on misread exits; the length gate drops it
                    paths[ring] = []
                    break
                tile = self.tiles.get((t, n))
                if tile is None:
                    logger.info("Knot walk left the mapped tiles at (%d,%d)", t, n)
                    break
                if paths[ring] and (t, n) == (paths[ring][0]["x"], paths[ring][0]["y"]):
                    break
                if tile["rune"] < 0:
                    rune = UNKNOWN
                elif tile["own"] == ring or tile["matched"]:
                    rune = tile["rune"]
                else:
                    rune = -1 - tile["rune"]
                paths[ring].append({"x": t, "y": n, "rune": rune})
                if tile["crossing"]:
                    hit = None
                    for inter in intersections:
                        if inter["x"] == t and inter["y"] == n:
                            hit = inter
                            break
                    if hit is None:
                        hit = {"x": t, "y": n, "col1": None, "i1": None, "col2": None, "i2": None}
                        intersections.append(hit)
                    if tile["own"] == ring:
                        hit["col1"], hit["i1"] = ring, len(paths[ring]) - 1
                    else:
                        hit["col2"], hit["i2"] = ring, len(paths[ring]) - 1
                exits = tile["exits"]
                moved = False
                for d, (dt, dn) in enumerate(DIRS):
                    if came_from == (d + 2) % 4:
                        continue
                    if exits[d] == ring:
                        t, n = t + dt, n + dn
                        came_from = d
                        moved = True
                        break
                if not moved:
                    break
            if len(paths[ring]) < 3:
                paths[ring] = []
        return paths, intersections


def read_normalized(norm: np.ndarray, assets: ClueAssets, scale: float = 1.0,
                    region: Tuple[int, int] = (0, 0),
                    origin: Optional[Tuple[int, int]] = None) -> Optional[KnotState]:
    """Read a knot from the 1x-normalized interface rect (KNOT_W x KNOT_H)."""
    if origin is not None:
        candidates = [origin]
    else:
        candidates = _Reader(norm, assets).origin_candidates()
        if not candidates:
            logger.info("Knot: no socket lattice found")
            return None

    best = None
    for cand in candidates:
        if best is not None and best[0] >= 60:
            break
        reader = _Reader(norm, assets)
        reader.pathor = cand
        if not reader._refine_origin():
            continue
        if reader.map_tiles():
            continue
        paths, intersections = reader.walk_paths()
        rings = [p for p in paths if p]
        if len(rings) < 2 or not intersections:
            logger.info(
                "Knot: origin %s mapped %d tiles but only %d rings / %d crossings",
                cand, len(reader.tiles), len(rings), len(intersections),
            )
            continue
        state = KnotState(
            paths=[[KnotSlot(x=s_["x"], y=s_["y"], rune=s_["rune"]) for s_ in p] for p in paths],
            intersections=intersections,
            rune_count=len(reader.rune_crops),
            scale=scale,
            region=region,
        )
        q = _state_quality(state)
        if best is None or q > best[0]:
            best = (q, state, reader)
    if best is None:
        return None
    _, state, reader = best
    logger.info(
        "Knot read: %d tiles, rings %s, %d crossings, %d distinct runes",
        len(reader.tiles),
        [len(p) for p in state.paths], len(state.intersections), state.rune_count,
    )
    return state


def _state_quality(state: Optional[KnotState]) -> int:
    if state is None:
        return -1
    cells = sum(len(p) for p in state.paths)
    known = sum(1 for p in state.paths for s in p if s.rune >= 0)
    return cells + known + 3 * len(state.intersections)


def read_knot(frame, assets: ClueAssets, hint: Optional[float] = None,
              debug_dir: Optional[object] = None) -> Optional[KnotState]:
    frame = detection.to_array(frame)
    x_match = detection.calibrate_scale(frame, assets.needle("exitbutton"), hint=hint)
    if not x_match.ok:
        logger.info("Knot: close button not found")
        return None
    # the anchor sprite is small, so its scale estimate is only good to a
    # couple percent; that drifts the 24px lattice by ~10px across the rect.
    # Read at candidate scales and keep the best-structured result.
    best = None
    last_norm = None
    for cs in (x_match.scale * f
               for f in (1.0, 0.992, 1.008, 0.985, 1.015, 0.977, 1.023, 0.97, 1.03)):
        if best is not None and best[0] >= 60:
            break
        x0 = int(round(x_match.x - (EOCX_OFFSET[0] + PAD) * cs))
        y0 = int(round(x_match.y - (EOCX_OFFSET[1] + PAD) * cs))
        w = int(round((KNOT_W + 2 * PAD) * cs))
        h = int(round((KNOT_H + 2 * PAD) * cs))
        if x0 < 0 or y0 < 0 or x0 + w > frame.shape[1] or y0 + h > frame.shape[0]:
            continue
        norm = detection.normalize_region(frame, x0, y0, w, h, cs)
        target = (KNOT_H + 2 * PAD, KNOT_W + 2 * PAD)
        if norm.shape[:2] != target:
            im = Image.fromarray(norm[:, :, :3].astype(np.uint8)).resize(
                (target[1], target[0]), Image.LANCZOS)
            norm = detection.to_array(np.asarray(im))
        norm = last_norm = norm[PAD:PAD + KNOT_H, PAD:PAD + KNOT_W]
        state = read_normalized(
            norm, assets, scale=cs, region=(x0 + int(PAD * cs), y0 + int(PAD * cs))
        )
        q = _state_quality(state)
        if q > 0 and (best is None or q > best[0]):
            best = (q, state)
    if last_norm is None:
        logger.info("Knot region out of frame")
        return None
    state = best[1] if best else None

    if state is None and debug_dir is not None:
        norm = last_norm
        try:
            path = str(debug_dir) + "/debug_knot_reject.png"
            Image.fromarray(norm[:, :, :3].astype(np.uint8)).save(path)
            logger.info("Saved rejected knot crop to %s", path)
        except Exception:
            pass
    return state


def merge_states(old: Optional[KnotState], new: KnotState) -> KnotState:
    """Fill unknown crossings in the new read from a previous read of the
    same knot (after the user inverted paths or rotated nothing). Rune ids
    are per-read discovery order, so the reads are aligned through the
    slots visible in both before comparing."""
    if old is None:
        return new
    if [len(p) for p in old.paths] != [len(p) for p in new.paths]:
        return new
    mapping = {}
    for ring in range(len(new.paths)):
        for i, slot in enumerate(new.paths[ring]):
            o, nr = old.paths[ring][i].rune, slot.rune
            if o >= 0 and nr >= 0:
                if mapping.get(nr, o) != o:
                    return new  # inconsistent; the board changed
                mapping[nr] = o
    if len(set(mapping.values())) != len(mapping):
        return new
    next_id = max([old.rune_count] + [v + 1 for v in mapping.values()])

    def remap(r):
        nonlocal next_id
        if r == UNKNOWN:
            return r
        k = -1 - r if r < 0 else r
        if k not in mapping:
            mapping[k] = next_id
            next_id += 1
        return -1 - mapping[k] if r < 0 else mapping[k]

    for ring in range(len(new.paths)):
        for slot in new.paths[ring]:
            slot.rune = remap(slot.rune)
    new.rune_count = next_id

    for ring in range(len(new.paths)):
        for i, slot in enumerate(new.paths[ring]):
            o = old.paths[ring][i]
            if slot.rune != o.rune:
                both_neg = slot.rune < 0 and o.rune < 0
                compat = (slot.rune < 0 and -1 - slot.rune != o.rune) or \
                         (o.rune < 0 and -1 - o.rune != slot.rune)
                if not (both_neg or compat):
                    return new  # different board; drop history
    for ring in range(len(new.paths)):
        for i, slot in enumerate(new.paths[ring]):
            o = old.paths[ring][i]
            if slot.rune < 0 <= o.rune:
                slot.rune = o.rune
    return new


def solve_knot(state: KnotState) -> List[KnotSolution]:
    paths = state.paths
    results = []

    def recurse(offsets: List[int], ring: int):
        length = max(1, len(paths[ring]))
        for off in range(length):
            offsets[ring] = off
            if ring + 1 < len(paths):
                recurse(offsets, ring + 1)
                continue
            sure = True
            ok = True
            for inter in state.intersections:
                c1, c2 = inter["col1"], inter["col2"]
                if c1 is None or c2 is None:
                    continue
                p = paths[c1][(inter["i1"] + offsets[c1]) % len(paths[c1])].rune
                f = paths[c2][(inter["i2"] + offsets[c2]) % len(paths[c2])].rune
                if p < 0 and f >= 0 and -p - 1 == f:
                    ok = False
                    break
                if f < 0 <= p and -f - 1 == p:
                    ok = False
                    break
                if p >= 0 and f >= 0 and p != f:
                    ok = False
                    break
                if p < 0 or f < 0:
                    sure = False
            if ok:
                results.append(KnotSolution(
                    state=state, offsets=offsets.copy(), sure=sure, candidates=0,
                ))
    recurse([0, 0, 0, 0], 0)
    for r in results:
        r.candidates = len(results)
    return results


def pick_solution(solutions: List[KnotSolution]) -> Optional[KnotSolution]:
    if len(solutions) == 1:
        return solutions[0]
    return next((s for s in solutions if s.sure), None)
