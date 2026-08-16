"""Slide puzzle board reading from a captured frame.

The slide interface is located by its close button sprite at any UI scale,
the 5x5 board region is normalized back to 1x geometry, and each tile is
identified by ZNCC against the runeapps reference tile sets. Small per-tile
position jitter absorbs residual scale error.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from runekit import detection
from .assets import SLIDE_THEMES, ClueAssets

logger = logging.getLogger(__name__)

TILE = 56  # on-screen tile size at 1x
BOARD_PX = TILE * 5
NEEDLE_TO_BOARD = (-297, 15)  # slide close button top-left -> board top-left
REF_TILE = 49  # reference art tile size
PAD = 8

THEME_PROBE_CELLS = (6, 12, 18)
BLANK = 24


@dataclass
class SlideBoard:
    theme: str
    board: List[int]
    scale: float
    board_rect: Tuple[int, int, int, int]  # frame coords: x, y, w, h
    confidence: float
    needle: detection.Match


@dataclass
class SlideSolution:
    board: SlideBoard
    moves: List[int]  # grid cells to click, in order


_BANKS = {}


def _theme_bank(assets: ClueAssets, theme: str) -> detection.TemplateBank:
    if theme not in _BANKS:
        img = assets.slide_theme(theme)
        tiles = [
            detection.to_array(img[r * REF_TILE:(r + 1) * REF_TILE, c * REF_TILE:(c + 1) * REF_TILE])
            for r in range(5)
            for c in range(5)
        ]
        _BANKS[theme] = detection.TemplateBank(tiles)
    return _BANKS[theme]


def _captured_tile(norm: np.ndarray, cell: int, jx: int = 0, jy: int = 0) -> Optional[np.ndarray]:
    r, c = divmod(cell, 5)
    x = PAD + c * TILE + jx
    y = PAD + r * TILE + jy
    crop = norm[y:y + TILE, x:x + TILE, :3]
    if crop.shape[:2] != (TILE, TILE):
        return None
    im = Image.fromarray(crop.astype(np.uint8)).resize((REF_TILE, REF_TILE), Image.LANCZOS)
    return detection.to_array(np.asarray(im))


def _normalize_board(frame: np.ndarray, needle: detection.Match, scale: float) -> Optional[np.ndarray]:
    bx = needle.x + NEEDLE_TO_BOARD[0] * scale
    by = needle.y + NEEDLE_TO_BOARD[1] * scale
    x0 = int(round(bx - PAD * scale))
    y0 = int(round(by - PAD * scale))
    size = int(round((BOARD_PX + 2 * PAD) * scale))
    if x0 < 0 or y0 < 0 or x0 + size > frame.shape[1] or y0 + size > frame.shape[0]:
        return None
    norm = detection.normalize_region(frame, x0, y0, size, size, scale)
    target = BOARD_PX + 2 * PAD
    if norm.shape[0] != target:
        im = Image.fromarray(norm[:, :, :3].astype(np.uint8)).resize((target, target), Image.LANCZOS)
        norm = detection.to_array(np.asarray(im))
    return norm


def read_slide(frame, assets: ClueAssets, hint: Optional[float] = None) -> Optional[SlideBoard]:
    frame = detection.to_array(frame)
    m = detection.calibrate_scale(frame, assets.needle("slide"), hint=hint)
    if not m.ok:
        return None
    logger.info("Slide interface at (%d, %d), scale %.3f, zncc %.3f", m.x, m.y, m.scale, m.zncc)

    # theme vote on probe cells, refining scale at the same time
    best = None  # (score, scale, norm, theme)
    for s in (m.scale * 0.95, m.scale, m.scale * 1.05):
        norm = _normalize_board(frame, m, s)
        if norm is None:
            continue
        caps = [c for c in (_captured_tile(norm, cell) for cell in THEME_PROBE_CELLS) if c is not None]
        if not caps:
            continue
        theme_scores = {}
        for theme in SLIDE_THEMES:
            bank = _theme_bank(assets, theme)
            theme_scores[theme] = sum(float(bank.scores(cap)[:BLANK].max()) for cap in caps)
        theme, score = max(theme_scores.items(), key=lambda kv: kv[1])
        if best is None or score > best[0]:
            best = (score, s, norm, theme)

    if best is None or best[0] / len(THEME_PROBE_CELLS) < 0.45:
        logger.info("Slide theme identification failed (best %.2f)", best[0] if best else -1)
        return None
    _, scale, norm, theme = best
    bank = _theme_bank(assets, theme)

    # blank cell: darkest, flattest tile
    stats = []
    for cell in range(25):
        cap = _captured_tile(norm, cell)
        stats.append((float(cap[:, :, :3].std()), float(cap[:, :, :3].mean()), cell))
    blank_cell = min(stats, key=lambda t: t[0] + t[1])[2]

    # score every remaining cell against every part with small jitter
    jitters = [(jx, jy) for jx in (-3, 0, 3) for jy in (-3, 0, 3)]
    cells = [c for c in range(25) if c != blank_cell]
    scores = np.full((25, 24), -1.0)
    for cell in cells:
        for jx, jy in jitters:
            cap = _captured_tile(norm, cell, jx, jy)
            if cap is None:
                continue
            scores[cell] = np.maximum(scores[cell], bank.scores(cap)[:BLANK])

    # greedy unique assignment
    board = [-1] * 25
    board[blank_cell] = BLANK
    taken_cells, taken_parts = {blank_cell}, set()
    pairs = sorted(
        ((scores[cell, part], cell, part) for cell in cells for part in range(24)),
        key=lambda t: -t[0],
    )
    assigned = []
    for score, cell, part in pairs:
        if cell in taken_cells or part in taken_parts:
            continue
        board[cell] = part
        taken_cells.add(cell)
        taken_parts.add(part)
        assigned.append(score)
        if len(taken_cells) == 25:
            break

    confidence = float(np.mean(assigned)) if assigned else 0.0
    if min(assigned) < 0.25 or confidence < 0.45:
        logger.info("Slide read rejected: min %.2f mean %.2f", min(assigned), confidence)
        return None

    bx = int(round(m.x + NEEDLE_TO_BOARD[0] * scale))
    by = int(round(m.y + NEEDLE_TO_BOARD[1] * scale))
    size = int(round(BOARD_PX * scale))
    return SlideBoard(
        theme=theme,
        board=board,
        scale=scale,
        board_rect=(bx, by, size, size),
        confidence=confidence,
        needle=m,
    )
