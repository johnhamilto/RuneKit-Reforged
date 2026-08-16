"""Towers puzzle: read the 5x5 skyscrapers board and solve it.

The interface is located from its close button and top-left corner sprites;
the twenty gold edge clues (and any white prefilled cells) are classified by
ZNCC against digit glyphs rendered from the game font over locally sampled
background. Solving is standard skyscrapers backtracking over row
permutations with column pruning.
"""
import itertools
import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from runekit import detection
from .assets import ClueAssets

logger = logging.getLogger(__name__)

INNER = 270  # inner board square at 1x, at (topleft.x + 20, exitbutton.y + 36)
GOLD = (255, 205, 10)
WHITE = (255, 255, 255)
PAD = 12
EDGE_MIN = 0.5  # required classification score for edge clues


@dataclass
class TowersRead:
    top: List[int]
    bot: List[int]
    left: List[int]
    right: List[int]
    filled: List[List[int]]  # 0 where empty
    scale: float
    inner_origin: Tuple[int, int]  # frame coords
    confidence: float


@dataclass
class TowersSolution:
    read: TowersRead
    grid: List[List[int]]
    solutions: int


class _DigitTemplates:
    """Digit glyphs 1..5 rendered as ink masks; composited per background."""

    def __init__(self, font: dict):
        self.h = font["height"]
        self.basey = font["basey"]
        shadow = font.get("shadow", False)
        step = 4 if shadow else 3
        self.glyphs = {}
        for ch in font["chars"]:
            if ch["chr"] not in "12345":
                continue
            px = np.array(ch["pixels"], dtype=np.float64).reshape(-1, step)
            self.glyphs[ch["chr"]] = (ch["width"], px)

    def classify(self, norm: np.ndarray, x: int, y: int, color,
                 jitter=range(-4, 5)) -> Tuple[int, float]:
        """Best digit at (x, y=baseline). Returns (digit or 0, score)."""
        col = np.asarray(color, dtype=np.float64)
        top = y - self.basey
        best_digit, best_score = 0, -1.0
        for chr_, (width, px) in self.glyphs.items():
            box_w = width + 2
            crop0 = norm[max(0, top - 6):top + self.h + 6, max(0, x - 6):x + box_w + 6, :3]
            if crop0.size == 0:
                continue
            bg = np.median(crop0.reshape(-1, 3), axis=0)
            t = np.zeros((self.h, box_w, 4), dtype=np.int32)
            t[:, :, :3] = bg.astype(np.int32)
            t[:, :, 3] = 255
            for gx, gy, alpha, *lum in px:
                if 0 <= gy < self.h and 0 <= gx < box_w:
                    p = alpha / 255.0
                    ink = col * (lum[0] / 255.0 if lum else 1.0)
                    t[int(gy), int(gx), :3] = (ink * p + t[int(gy), int(gx), :3] * (1 - p)).astype(np.int32)
            for jx in jitter:
                for jy in jitter:
                    score = detection.zncc(norm, t, x + jx, top + jy)
                    if score > best_score:
                        best_score, best_digit = score, int(chr_)
        return best_digit, best_score


def read_towers(frame, assets: ClueAssets, hint: Optional[float] = None) -> Optional[TowersRead]:
    frame = detection.to_array(frame)
    exitbtn = detection.to_array(assets.needle("exitbutton"))
    topleft = detection.to_array(assets.needle("topleft"))
    digits = _DigitTemplates(assets.font("font_chat14"))

    x_match = detection.calibrate_scale(frame, exitbtn, hint=hint)
    if not x_match.ok:
        logger.info("Towers: close button not found")
        return None
    s = x_match.scale

    # the top-left corner sprite sits level with the close button
    x0 = max(0, int(x_match.x - 900 * s))
    y0 = max(0, int(x_match.y - 60 * s))
    y1 = min(frame.shape[0], int(x_match.y + 60 * s))
    sub = frame[y0:y1, x0:x_match.x]
    corner = detection.find_template(sub, topleft, np.linspace(0.94, 1.06, 7) * s)
    if corner.zncc < 0.5:
        logger.info("Towers: corner not found (zncc %.2f)", corner.zncc)
        return None
    corner_x = x0 + corner.x
    logger.info("Towers modal: X at (%d,%d), corner at %d, scale %.3f", x_match.x, x_match.y, corner_x, s)

    inner_x = corner_x + 20 * s
    inner_y = x_match.y + 36 * s

    best = None  # (confidence, read)
    for cs in (s * 0.95, s, s * 1.05):
        size = int(round((INNER + 2 * PAD) * cs))
        rx = int(round(inner_x - PAD * cs))
        ry = int(round(inner_y - PAD * cs))
        if rx < 0 or ry < 0 or rx + size > frame.shape[1] or ry + size > frame.shape[0]:
            continue
        norm = detection.normalize_region(frame, rx, ry, size, size, cs)
        target = INNER + 2 * PAD
        if norm.shape[0] != target:
            im = Image.fromarray(norm[:, :, :3].astype(np.uint8)).resize((target, target), Image.LANCZOS)
            norm = detection.to_array(np.asarray(im))

        def edge(x, y):
            return digits.classify(norm, PAD + x, PAD + y, GOLD)

        reads = []
        scores = []
        for a in range(5):
            reads.append(edge(43 + 44 * a, 14))
        for a in range(5):
            reads.append(edge(43 + 44 * a, 264))
        for a in range(5):
            reads.append(edge(6, 51 + 44 * a))
        for a in range(5):
            reads.append(edge(256, 51 + 44 * a))
        scores = [sc for _, sc in reads]
        conf = float(np.mean(scores))
        if best is None or conf > best[0]:
            filled = [[0] * 5 for _ in range(5)]
            for r in range(5):
                for c in range(5):
                    d, sc = digits.classify(norm, PAD + 43 + 44 * c, PAD + 50 + 44 * r, WHITE)
                    if sc >= 0.6:
                        filled[r][c] = d
            read = TowersRead(
                top=[d for d, _ in reads[0:5]],
                bot=[d for d, _ in reads[5:10]],
                left=[d for d, _ in reads[10:15]],
                right=[d for d, _ in reads[15:20]],
                filled=filled,
                scale=cs,
                inner_origin=(int(round(inner_x)), int(round(inner_y))),
                confidence=conf,
            )
            best = (conf, min(scores), read)

    if best is None:
        return None
    conf, min_score, read = best
    if min_score < 0.35 or conf < EDGE_MIN:
        logger.info("Towers read rejected: min %.2f mean %.2f", min_score, conf)
        return None
    return read


def _visible(seq: Sequence[int]) -> int:
    count, mx = 0, 0
    for v in seq:
        if v > mx:
            count += 1
            mx = v
    return count


def solve_towers(read: TowersRead, limit=2) -> List[List[List[int]]]:
    perms = list(itertools.permutations(range(1, 6)))
    row_cands = []
    for r in range(5):
        cands = []
        for p in perms:
            if read.left[r] and _visible(p) != read.left[r]:
                continue
            if read.right[r] and _visible(p[::-1]) != read.right[r]:
                continue
            if any(read.filled[r][c] and p[c] != read.filled[r][c] for c in range(5)):
                continue
            cands.append(p)
        row_cands.append(cands)

    solutions = []

    def place(r, rows):
        if len(solutions) >= limit:
            return
        if r == 5:
            for c in range(5):
                col = [rows[i][c] for i in range(5)]
                if read.top[c] and _visible(col) != read.top[c]:
                    return
                if read.bot[c] and _visible(col[::-1]) != read.bot[c]:
                    return
            solutions.append([list(p) for p in rows])
            return
        for p in row_cands[r]:
            ok = True
            for c in range(5):
                seen = [rows[i][c] for i in range(r)]
                if p[c] in seen:
                    ok = False
                    break
                # partial top-visibility pruning
                if read.top[c] and r == 4:
                    pass
            if ok:
                place(r + 1, rows + [p])

    place(0, [])
    return solutions
