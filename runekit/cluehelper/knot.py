"""Celtic knot puzzle: read the rings, solve the rotations.

Reader for scaled captures. The interface is normalized to 1x from the
close button. Tiles are found by their rune discs on a 24px isometric
lattice, ring membership comes from the ribbon fill sampled around each
disc, and the paths come from structure: crossings pass straight through,
every other tile joins exactly two same-ring neighbours, and where more
neighbours qualify the dark outline between side-by-side ribbons tells
joined tiles from adjacent ones. Runes are identified by clustering tile
crops against each other, so no reference rune art is needed. Crossings
hidden under the other ring are constrained by the match/mismatch border
and can be filled in by merging a second read after the user inverts paths.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from runekit import detection
from .assets import ClueAssets

logger = logging.getLogger(__name__)

KNOT_W, KNOT_H = 504, 326
EOCX_OFFSET = (483, 8)  # close button position within the knot rect
STEP = 24
MAX_TILES = 90  # real knots reach ~50 cells
PAD = 8

RING_NAMES = {0: "blue", 1: "gold", 2: "black", 3: "silver"}
RING_LETTERS = {"B": 0, "G": 1, "K": 2, "S": 3}

RUNE_SAME = 0.8  # mean-centred glyph correlation above which two runes match
GLYPH_SHIFT = 3  # alignment search around the rune area, px
BORDER_MIN = 0.6  # border sprite ZNCC to accept a crossing

# tile geometry at 1x: the grey rune disc (radius ~12) sits at the tile
# centre, the match/nomatch border ring hugs it (radius 13-17), and the
# ribbons are ~33px wide bands along the lattice diagonals
DISC_R = 11  # disc kernel radius for tile detection
DISC_MIN = 0.35  # disc response that still counts as a tile (background ~0.2)
RIM_MIN = 0.5  # fallback: grey rim coverage when a large glyph eats the disc
ARC_R = 20  # fill samples: outside the border ring, inside the band overlap
LINK_MIN = 25  # darkest cross-section between joined discs; an outline reads ~10-20

# lattice steps: dir 0..3 = (t,n+1) up-left, (t+1,n) up-right,
# (t,n-1) down-right, (t-1,n) down-left
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


def _ring_of(rgb: np.ndarray) -> Optional[str]:
    """Ring letter for one ribbon-fill sample, or None for parchment, disc
    and blends. Gold is bright and saturated. Black fill (~(16,16,22)) is
    told from blue (~(40,40,65)) by brightness first: black is faintly
    blue-tinted itself. Silver is unsupported until a capture shows one."""
    total = float(rgb.sum())
    if total <= 0:
        return None
    r, g, b = (float(v) for v in rgb)
    if r * 255 / total > 115 and b * 255 / total < 45 and total > 120:
        return "G"
    if total > 300 or r - b > 25:
        return None
    if total < 80:
        return "K"
    if b - r >= 8:
        return "B"
    return None


class _Reader:
    def __init__(self, norm: np.ndarray, assets: ClueAssets):
        self.norm = norm
        self.border_match = detection.to_array(assets.needle("bordermatch"))
        self.border_nomatch = detection.to_array(assets.needle("bordernomatch"))
        runearea = detection.to_array(assets.needle("runearea"))
        self.rune_mask = runearea[:, :, 3] > 128
        self.rune_size = runearea.shape[:2]
        self.tiles: Dict[Tuple[int, int], dict] = {}
        self.links: Dict[Tuple[Tuple[int, int], int], bool] = {}
        self.glyphs: Dict[Tuple[int, int], np.ndarray] = {}
        self.rune_count = 0
        self.pitch = float(STEP)  # measured lattice pitch, set by find_tiles
        rgb = norm[:, :, :3]
        self._sum = rgb.sum(axis=2)
        self._lowsat = (np.abs(rgb[..., 0] - rgb[..., 1]) < 30) & (
            np.abs(rgb[..., 1] - rgb[..., 2]) < 50
        )

    def find_tiles(self) -> Optional[str]:
        """Tiles by their rune discs: a round, low-saturation, bright blob
        on every tile. Score pixels by disc-kernel coverage, pick the lattice
        phase that lines up the most discs, keep the cells that respond."""
        h, w = self._sum.shape
        yy, xx = np.mgrid[-DISC_R:DISC_R + 1, -DISC_R:DISC_R + 1]
        rr = xx ** 2 + yy ** 2
        disc_k = (rr <= DISC_R ** 2).astype(np.float32)
        rim_k = ((rr <= DISC_R ** 2) & (rr >= 49)).astype(np.float32)
        score = cv2.filter2D(
            ((self._sum > 200) & self._lowsat).astype(np.float32), -1,
            disc_k / disc_k.sum(), borderType=cv2.BORDER_CONSTANT,
        )
        rim = cv2.filter2D(
            ((self._sum > 150) & self._lowsat).astype(np.float32), -1,
            rim_k / rim_k.sum(), borderType=cv2.BORDER_CONSTANT,
        )
        # the isometric lattice is two square lattices of pitch 2*STEP
        # offset by (STEP, STEP): fold strong responses onto one period
        period = 2 * STEP
        strong = np.where(score > 0.6, score, 0.0).astype(np.float32)
        ph, pw = -(-h // period) * period, -(-w // period) * period
        folded = np.zeros((ph, pw), np.float32)
        folded[:h, :w] = strong
        phase = folded.reshape(ph // period, period, pw // period, period).sum(axis=(0, 2))
        phase = phase + np.roll(phase, (-STEP, -STEP), axis=(0, 1))
        py0, px0 = np.unravel_index(int(phase.argmax()), phase.shape)
        if phase[py0, px0] <= 0:
            return "no rune discs found"
        self.tiles = {}
        margin = ARC_R + 3
        for j in range(-2, h // STEP + 2):
            for i in range(-2, w // STEP + 2):
                if (i + j) % 2:
                    continue
                x, y = px0 + STEP * i, py0 + STEP * j
                if not (margin <= x < w - margin and margin <= y < h - margin):
                    continue
                win = score[y - 3:y + 4, x - 3:x + 4]
                dy, dx = np.unravel_index(int(win.argmax()), win.shape)
                cx, cy = x + dx - 3, y + dy - 3
                if score[cy, cx] < DISC_MIN and rim[cy, cx] < RIM_MIN:
                    continue
                self.tiles[((i - j) // 2, (-i - j) // 2)] = {"cx": int(cx), "cy": int(cy)}
        if not self.tiles:
            return "no rune discs found"
        if len(self.tiles) > MAX_TILES:
            return f"{len(self.tiles)} tiles found; the rings could not be mapped"
        # measured lattice pitch: with ~40 discs this pins the interface
        # scale far better than the close button sprite does
        keys = np.array(list(self.tiles), dtype=float)
        centres = np.array([(t["cx"], t["cy"]) for t in self.tiles.values()], dtype=float)
        u = keys[:, 0] - keys[:, 1]
        v = -(keys[:, 0] + keys[:, 1])
        uc, vc = u - u.mean(), v - v.mean()
        denom = float(uc @ uc + vc @ vc)
        if denom > 0:
            cx, cy = centres[:, 0], centres[:, 1]
            self.pitch = float(uc @ (cx - cx.mean()) + vc @ (cy - cy.mean())) / denom
        return None

    def classify(self) -> Optional[str]:
        """Per tile: crossing and match flags from the border sprites, ring
        from the ribbon fill at the four edge midpoints (outside the border
        ring, inside the overlap of the tile's two possible bands, so at a
        crossing it is the ring drawn on top), and the glyph crop."""
        rgb = self.norm[:, :, :3]
        f32 = rgb.astype(np.float32)
        sqdiff = {
            name: np.nan_to_num(detection._masked_sqdiff(f32, spr), nan=np.inf)
            for name, spr in (("m", self.border_match), ("n", self.border_nomatch))
        }
        for key, tile in self.tiles.items():
            cx, cy = tile["cx"], tile["cy"]
            sx, sy = cx - 23, cy - 23
            tile["crossing"], tile["matched"] = self._border(sx, sy, sqdiff)
            votes = []
            for dx, dy in ((0, -ARC_R), (0, ARC_R), (ARC_R, 0), (-ARC_R, 0)):
                patch = rgb[cy + dy - 1:cy + dy + 2, cx + dx - 2:cx + dx + 3].reshape(-1, 3)
                letter = _ring_of(patch.mean(axis=0))
                if letter:
                    votes.append(letter)
            tile["ring"] = min(set(votes), key=lambda l: (-votes.count(l), l)) if votes else None
            tile["rune"] = -1
            glyph = self._glyph(sx, sy)
            if glyph is not None:
                self.glyphs[key] = glyph
        # a grey blob with no ribbon around it (a button, a stray match) is
        # not a tile; dropping it keeps a real neighbour from a false loose end
        self.tiles = {
            key: tile for key, tile in self.tiles.items()
            if tile["ring"] is not None or tile["crossing"]
        }
        return None

    def _border(self, sx: int, sy: int, sqdiff: dict) -> Tuple[bool, bool]:
        """The border sprite sits at tile origin + (6, 6); take the best
        SQDIFF spot within a few pixels and score it by ZNCC."""
        best = {}
        for name, spr in (("m", self.border_match), ("n", self.border_nomatch)):
            res = sqdiff[name]
            x0, y0 = max(0, sx + 6 - 3), max(0, sy + 6 - 3)
            win = res[y0:y0 + 7, x0:x0 + 7]
            if win.size == 0:
                best[name] = -1.0
                continue
            dy, dx = np.unravel_index(int(win.argmin()), win.shape)
            best[name] = detection.zncc(self.norm, spr, x0 + dx, y0 + dy)
        if max(best.values()) < BORDER_MIN:
            return False, False
        return True, best["m"] >= best["n"]

    def _glyph(self, sx: int, sy: int) -> Optional[np.ndarray]:
        """Masked RGB values of the rune area at every alignment within
        GLYPH_SHIFT of the nominal spot: (2J+1, 2J+1, 3 * mask px)."""
        h, w = self.rune_size
        J = GLYPH_SHIFT
        crop = self.norm[sy + 11 - J:sy + 11 + h + J, sx + 11 - J:sx + 11 + w + J, :3]
        if crop.shape[:2] != (h + 2 * J, w + 2 * J):
            return None
        windows = np.lib.stride_tricks.sliding_window_view(crop, (h, w), axis=(0, 1))
        return windows[:, :, :, self.rune_mask].reshape(2 * J + 1, 2 * J + 1, -1).astype(np.float32)

    def identify_runes(self) -> Optional[str]:
        """Cluster the glyph crops against each other, so no reference rune
        art is needed. Correlation runs on masked RGB after subtracting the
        board's mean crop, which removes the disc shading every tile shares
        and leaves the glyph; colour is part of the signature because some
        runes share a shape. The alignment search absorbs lattice jitter."""
        J = GLYPH_SHIFT
        keys = [k for k in sorted(self.tiles) if k in self.glyphs]
        if not keys:
            return "no rune glyphs readable"
        centre = np.mean([self.glyphs[k][J, J] for k in keys], axis=0)
        reps: List[Tuple[np.ndarray, np.ndarray]] = []  # (centre vector, all shifts)
        for key in keys:
            v = self.glyphs[key] - centre
            v = v - v.mean(axis=-1, keepdims=True)
            v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
            for idx, (rep_centre, rep_shifts) in enumerate(reps):
                if max((v @ rep_centre).max(), (rep_shifts @ v[J, J]).max()) >= RUNE_SAME:
                    self.tiles[key]["rune"] = idx
                    break
            else:
                if len(reps) < 30:
                    reps.append((v[J, J], v))
                    self.tiles[key]["rune"] = len(reps) - 1
        self.rune_count = len(reps)
        return None

    def _link_score(self, a: dict, b: dict) -> float:
        """Darkest cross-section of the ribbon between two adjacent discs,
        averaged across the band so weave texture doesn't fake an outline.
        Joined tiles share continuous fill; side-by-side ribbons meet at a
        dark outline about a third of the way between the centres."""
        c1 = np.array((a["cx"], a["cy"]), dtype=float)
        c2 = np.array((b["cx"], b["cy"]), dtype=float)
        axis = (c2 - c1) / np.linalg.norm(c2 - c1)
        perp = np.array((-axis[1], axis[0]))
        ks = np.arange(13.0, 21.5, 0.5)[:, None, None]
        ps = np.array([-8.0, -4.0, 0.0, 4.0, 8.0])[None, :, None]
        pts = np.rint(c1 + axis * ks + perp * ps).astype(int)
        h, w = self._sum.shape
        xs = np.clip(pts[..., 0], 0, w - 1)
        ys = np.clip(pts[..., 1], 0, h - 1)
        return float(self._sum[ys, xs].mean(axis=1).min())

    def link(self) -> Optional[str]:
        """Decide which neighbours each tile's ribbon continues to. Crossings
        join all four: two rings passing straight through. Other tiles join
        exactly two same-ring neighbours, best cross-section first."""
        cands = []
        for key, tile in self.tiles.items():
            for d in (0, 1):
                nb = (key[0] + DIRS[d][0], key[1] + DIRS[d][1])
                other = self.tiles.get(nb)
                if other is None:
                    continue
                if tile["crossing"] or other["crossing"]:
                    cands.append((float("inf"), key, nb, d))
                elif tile["ring"] == other["ring"]:
                    cands.append((self._link_score(tile, other), key, nb, d))
        cands.sort(key=lambda c: -c[0])
        capacity = {key: 4 if tile["crossing"] else 2 for key, tile in self.tiles.items()}
        self.links = {}
        for score, key, nb, d in cands:
            if score < LINK_MIN or not capacity[key] or not capacity[nb]:
                continue
            self.links[(key, d)] = self.links[(nb, (d + 2) % 4)] = True
            capacity[key] -= 1
            capacity[nb] -= 1
        loose = sum(1 for c in capacity.values() if c)
        if loose:
            return f"{loose} of {len(self.tiles)} tiles have a loose end"
        return None

    def walk(self):
        """Follow each ring from its highest-keyed plain tile: along the
        single other link on plain tiles, straight through crossings."""
        paths = [[] for _ in range(4)]
        intersections = []
        seen: Dict[Tuple[int, int], int] = {}
        for letter, ring in sorted(RING_LETTERS.items(), key=lambda kv: kv[1]):
            name = RING_NAMES[ring]
            own = [k for k, t in self.tiles.items() if t["ring"] == letter and not t["crossing"]]
            if not own:
                continue
            start = max(own, key=lambda k: 100 * k[0] + k[1])
            d = next((dd for dd in range(4) if self.links.get((start, dd))), None)
            if d is None:
                return None, None, f"{name} ring start tile has no links"
            cur = start
            while True:
                tile = self.tiles[cur]
                if not tile["crossing"] and tile["ring"] != letter:
                    return None, None, f"{name} ring runs into a {tile['ring']} tile at {cur}"
                if tile["rune"] < 0:
                    rune = UNKNOWN
                elif tile["ring"] == letter or tile["matched"]:
                    rune = tile["rune"]
                else:
                    rune = -1 - tile["rune"]
                paths[ring].append({"x": cur[0], "y": cur[1], "rune": rune})
                seen[cur] = seen.get(cur, 0) + 1
                if tile["crossing"]:
                    hit = next((i for i in intersections if (i["x"], i["y"]) == cur), None)
                    if hit is None:
                        hit = {"x": cur[0], "y": cur[1], "col1": None, "i1": None, "col2": None, "i2": None}
                        intersections.append(hit)
                    if tile["ring"] == letter:
                        hit["col1"], hit["i1"] = ring, len(paths[ring]) - 1
                    else:
                        hit["col2"], hit["i2"] = ring, len(paths[ring]) - 1
                elif cur != start:
                    outs = [dd for dd in range(4) if self.links.get((cur, dd)) and dd != (d + 2) % 4]
                    if len(outs) != 1:
                        return None, None, f"{name} ring forks at {cur}"
                    d = outs[0]
                nxt = (cur[0] + DIRS[d][0], cur[1] + DIRS[d][1])
                if nxt == start:
                    break
                if nxt not in self.tiles or len(paths[ring]) > MAX_TILES:
                    return None, None, f"{name} ring does not close"
                cur = nxt
        for key, tile in self.tiles.items():
            want = 2 if tile["crossing"] else 1
            if seen.get(key, 0) != want:
                return None, None, f"tile {key} lies on {seen.get(key, 0)} rings, expected {want}"
        if any(i["col1"] is None or i["col2"] is None for i in intersections):
            return None, None, "a crossing's top ring matches neither ring through it"
        return paths, intersections, None


def read_normalized(norm: np.ndarray, assets: ClueAssets, scale: float = 1.0,
                    region: Tuple[int, int] = (0, 0)) -> Optional[KnotState]:
    """Read a knot from the 1x-normalized interface rect (KNOT_W x KNOT_H)."""
    reader = _Reader(norm, assets)
    for step in (reader.find_tiles, reader.classify, reader.link, reader.identify_runes):
        err = step()
        if err:
            logger.info("Knot: %s", err)
            return None
    paths, intersections, err = reader.walk()
    if err:
        logger.info(
            "Knot: %s (%d tiles, %d crossings)", err, len(reader.tiles),
            sum(1 for t in reader.tiles.values() if t["crossing"]),
        )
        return None
    rings = [p for p in paths if p]
    if len(rings) < 2 or not intersections:
        logger.info(
            "Knot: %d tiles but only %d rings / %d crossings",
            len(reader.tiles), len(rings), len(intersections),
        )
        return None
    state = KnotState(
        paths=[[KnotSlot(x=s_["x"], y=s_["y"], rune=s_["rune"]) for s_ in p] for p in paths],
        intersections=intersections,
        rune_count=reader.rune_count,
        scale=scale * reader.pitch / STEP,
        region=region,
    )
    logger.info(
        "Knot read: %d tiles, rings %s, %d crossings, %d distinct runes, pitch %.2f",
        len(reader.tiles),
        [len(p) for p in state.paths], len(state.intersections), state.rune_count,
        reader.pitch,
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
    last_norm = None

    def attempt(cs: float):
        nonlocal last_norm
        x0 = int(round(x_match.x - (EOCX_OFFSET[0] + PAD) * cs))
        y0 = int(round(x_match.y - (EOCX_OFFSET[1] + PAD) * cs))
        w = int(round((KNOT_W + 2 * PAD) * cs))
        h = int(round((KNOT_H + 2 * PAD) * cs))
        if x0 < 0 or y0 < 0 or x0 + w > frame.shape[1] or y0 + h > frame.shape[0]:
            return None
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
        return (q, state, cs) if q > 0 else None

    # the anchor sprite is small, so its scale estimate is only good to a
    # couple percent; that drifts the 24px lattice by ~10px across the rect.
    # Read at candidate scales until one is well structured.
    best = None
    for f in (1.0, 0.992, 1.008, 0.985, 1.015, 0.977, 1.023, 0.97, 1.03):
        if best is not None and best[0] >= 60:
            break
        got = attempt(x_match.scale * f)
        if got and (best is None or got[0] > best[0]):
            best = got
    if last_norm is None:
        logger.info("Knot region out of frame")
        return None
    # the read reports the scale the disc lattice measured; a re-read at
    # that scale lines the rune crops up, which the clustering needs
    if best is not None and abs(best[1].scale / best[2] - 1) > 0.003:
        got = attempt(best[1].scale)
        if got and got[0] >= best[0]:
            best = got
    state = best[1] if best else None

    if state is None and debug_dir is not None:
        norm = last_norm
        try:
            path = str(debug_dir) + "/debug_knot_reject.png"
            Image.fromarray(norm[:, :, :3].astype(np.uint8)).save(path)
            half = Image.fromarray(frame[:, :, :3].astype(np.uint8)).reduce(2)
            half.convert("RGB").save(str(debug_dir) + "/debug_knot_frame.jpg", quality=70)
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
