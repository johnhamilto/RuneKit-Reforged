"""Lockbox puzzle: read the 5x5 melee/range/mage grid and compute presses.

Clicking a tile advances it and its orthogonal neighbours by one (mod 3).
The box unlocks when every tile shows the same symbol. Solved by chasing
lights row by row for each of the 243 top-row press combinations and each
target symbol, keeping the press map with the fewest total presses.

The grid is located by finding one tile sprite at any UI scale, then testing
candidate grid origins around it; no modal geometry is needed.
"""
import itertools
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from runekit import detection
from .assets import ClueAssets

logger = logging.getLogger(__name__)

TILE = 38  # tile stride at 1x
TILE_NAMES = ("melee", "range", "mage")


@dataclass
class LockboxRead:
    grid: List[List[int]]  # [row][col] -> 0 melee, 1 range, 2 mage
    scale: float
    origin: Tuple[int, int]  # frame coords of grid top-left
    confidence: float


@dataclass
class LockboxSolution:
    read: LockboxRead
    presses: List[List[int]]  # [row][col] -> presses (0..2)
    target: int


PAD = 10


def _normalize_grid(frame, ox: float, oy: float, scale: float) -> Optional[np.ndarray]:
    x0 = int(round(ox - PAD * scale))
    y0 = int(round(oy - PAD * scale))
    size = int(round((5 * TILE + 2 * PAD) * scale))
    if x0 < 0 or y0 < 0 or x0 + size > frame.shape[1] or y0 + size > frame.shape[0]:
        return None
    norm = detection.normalize_region(frame, x0, y0, size, size, scale)
    target = 5 * TILE + 2 * PAD
    if norm.shape[0] != target:
        im = Image.fromarray(norm[:, :, :3].astype(np.uint8)).resize((target, target), Image.LANCZOS)
        norm = detection.to_array(np.asarray(im))
    return norm


def _classify_cell(norm, r: int, c: int, needles, jitter=(-3, 0, 3)) -> List[float]:
    best = [-1.0] * len(needles)
    for jx in jitter:
        for jy in jitter:
            x0 = PAD + c * TILE + jx
            y0 = PAD + r * TILE + jy
            cap = norm[y0:y0 + TILE, x0:x0 + TILE]
            if cap.shape[:2] != (TILE, TILE):
                continue
            for i, needle in enumerate(needles):
                best[i] = max(best[i], detection.zncc(cap, needle, 0, 0))
    return best


def _find_in_region(frame, needle, x0, y0, x1, y1, scales):
    """find_template restricted to a frame region; coordinates mapped back."""
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(frame.shape[1], x1), min(frame.shape[0], y1)
    if x1 - x0 < 80 or y1 - y0 < 80:
        return None
    sub = frame[y0:y1, x0:x1]
    m = detection.find_template(sub, needle, scales)
    m.x += x0
    m.y += y0
    return m


def read_lockbox(frame, assets: ClueAssets) -> Optional[LockboxRead]:
    frame = detection.to_array(frame)
    needles = [detection.to_array(assets.needle(n)) for n in TILE_NAMES]

    anchor = None
    anchor_needle = None
    for needle in needles:
        m = detection.calibrate_scale(frame, needle)
        if m.ok and (anchor is None or m.zncc > anchor.zncc):
            anchor = m
            anchor_needle = needle
    if anchor is None:
        return None
    logger.info(
        "Lockbox tile at (%d, %d), scale %.3f, zncc %.3f",
        anchor.x, anchor.y, anchor.scale, anchor.zncc,
    )

    # Refine the noisy scale estimate: locate a second tile of a different
    # symbol near the anchor; the displacement must be an integer number of
    # 38px strides, which pins the true scale.
    s = anchor.scale
    reach = int(round((4 * TILE + 30) * s))
    fine = np.linspace(0.94, 1.06, 7) * s
    refinements = []
    for needle in needles:
        if needle is anchor_needle:
            continue
        m = _find_in_region(
            frame, needle,
            anchor.x - reach, anchor.y - reach,
            anchor.x + reach + int(TILE * s), anchor.y + reach + int(TILE * s),
            fine,
        )
        if m is None or m.zncc < 0.6:
            continue
        d = m.x - anchor.x if abs(m.x - anchor.x) >= abs(m.y - anchor.y) else m.y - anchor.y
        steps = round(d / (TILE * s))
        if steps == 0:
            continue
        cand = d / (TILE * steps)
        if abs(cand / s - 1) < 0.1 and abs(d / (TILE * cand) - steps) < 0.15:
            refinements.append((abs(steps), m.zncc, cand))
    candidate_scales = [s]
    if refinements:
        # the farthest displacement has the smallest quantization error
        refinements.sort(key=lambda t: (-t[0], -t[1]))
        candidate_scales = [refinements[0][2], refinements[0][2] * 0.99,
                            refinements[0][2] * 1.01, s]
        logger.info(
            "Lockbox scale candidates %.4f (over %d strides) and %.4f",
            refinements[0][2], refinements[0][0], s,
        )
    s = candidate_scales[0]

    # tile background color: pixels near the anchor where the sprite art is
    # transparent, so classification can use dense composited templates
    size = int(round(TILE * s))
    tile_crop = frame[anchor.y:anchor.y + size, anchor.x:anchor.x + size, :3]
    if tile_crop.shape[0] != size or tile_crop.shape[1] != size:
        return None
    alpha = np.asarray(
        Image.fromarray((anchor_needle[:, :, 3]).astype(np.uint8)).resize((size, size), Image.NEAREST)
    )
    bg_px = tile_crop[alpha < 16]
    bg = np.median(bg_px, axis=0) if len(bg_px) > 30 else np.median(tile_crop.reshape(-1, 3), axis=0)
    dense = []
    for needle in needles:
        a = needle[:, :, 3:4] / 255.0
        comp = np.zeros((TILE, TILE, 4), dtype=np.int32)
        comp[:, :, :3] = (needle[:, :, :3] * a + bg * (1 - a)).astype(np.int32)
        comp[:, :, 3] = 255
        dense.append(comp)

    # the anchor's grid index is unknown; per candidate scale, pick the best
    # of the 25 possible origins by probe cells, fully classify that grid,
    # and keep the scale whose full classification is most confident
    probes = ((0, 0), (0, 4), (4, 0), (4, 4), (2, 2))
    best = None  # (confidence, min_score, grid, origin, scale)
    for cs in candidate_scales:
        stride = TILE * cs
        origin_best = None  # (probe score, norm, origin)
        for r in range(5):
            for c in range(5):
                norm = _normalize_grid(frame, anchor.x - c * stride, anchor.y - r * stride, cs)
                if norm is None:
                    continue
                score = sum(max(_classify_cell(norm, rr, cc, dense)) for rr, cc in probes)
                if origin_best is None or score > origin_best[0]:
                    origin_best = (score, norm, (anchor.x - c * stride, anchor.y - r * stride))
        if origin_best is None or origin_best[0] / len(probes) < 0.4:
            continue
        _, norm, origin = origin_best

        grid = []
        scores = []
        for r in range(5):
            row = []
            for c in range(5):
                z = _classify_cell(norm, r, c, dense, jitter=range(-6, 7))
                row.append(int(np.argmax(z)))
                scores.append(max(z))
            grid.append(row)
        conf = float(np.mean(scores))
        if best is None or conf > best[0]:
            best = (conf, float(min(scores)), grid, origin, cs)

    if best is None:
        logger.info("Lockbox grid origin not found")
        return None
    confidence, min_score, grid, (ox, oy), s = best
    if min_score < 0.28 or confidence < 0.55:
        logger.info("Lockbox read rejected: min %.2f mean %.2f", min_score, confidence)
        return None
    return LockboxRead(
        grid=grid,
        scale=s,
        origin=(int(round(ox)), int(round(oy))),
        confidence=confidence,
    )


def _press(grid: List[List[int]], presses: List[List[int]], r: int, c: int, times: int):
    presses[r][c] = (presses[r][c] + times) % 3
    for rr, cc in ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= rr < 5 and 0 <= cc < 5:
            grid[rr][cc] = (grid[rr][cc] + times) % 3


def solve_lockbox(grid: List[List[int]]) -> Optional[Tuple[List[List[int]], int]]:
    """Fewest-press map making every tile equal, or None if unreadable."""
    best = None
    for target in range(3):
        for top in itertools.product(range(3), repeat=5):
            g = [row[:] for row in grid]
            presses = [[0] * 5 for _ in range(5)]
            for c, times in enumerate(top):
                _press(g, presses, 0, c, times)
            for r in range(1, 5):
                for c in range(5):
                    need = (target - g[r - 1][c]) % 3
                    _press(g, presses, r, c, need)
            if all(g[4][c] == target for c in range(5)):
                total = sum(map(sum, presses))
                if best is None or total < best[0]:
                    best = (total, presses, target)
    if best is None:
        return None
    return best[1], best[2]
