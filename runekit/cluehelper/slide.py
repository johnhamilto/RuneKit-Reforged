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
PAD = 36  # normalized margin; leaves room to search for the true board origin

THEME_PROBE_CELLS = (6, 12, 18)
BLANK = 24
ORIGIN_SWEEP = 28  # board origin search radius around the nominal offset


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
    if x < 0 or y < 0:
        return None
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


def _probe_score(assets: ClueAssets, norm: np.ndarray, theme: str,
                 dx: int = 0, dy: int = 0) -> float:
    bank = _theme_bank(assets, theme)
    caps = [
        c for c in (_captured_tile(norm, cell, dx, dy) for cell in THEME_PROBE_CELLS)
        if c is not None
    ]
    if not caps:
        return -1.0
    total = 0.0
    for cap in caps:
        # near-flat crops (plain interface panels) make ZNCC degenerate and
        # can score high against the darkest themes; they prove nothing
        if float(cap[:, :, :3].std()) < 8:
            continue
        total += float(bank.scores(cap)[:BLANK].max())
    return total / len(caps)


def _classify_board(assets: ClueAssets, norm: np.ndarray, theme: str, dx: int, dy: int):
    """Full board read at an origin offset: (board, min score, mean score)."""
    bank = _theme_bank(assets, theme)

    stats = []
    for cell in range(25):
        cap = _captured_tile(norm, cell, dx, dy)
        if cap is None:
            return None
        stats.append((float(cap[:, :, :3].std()) + float(cap[:, :, :3].mean()), cell))
    blank_cell = min(stats)[1]

    jitters = [(jx, jy) for jx in (-3, 0, 3) for jy in (-3, 0, 3)]
    cells = [c for c in range(25) if c != blank_cell]
    scores = np.full((25, 24), -1.0)
    for cell in cells:
        for jx, jy in jitters:
            cap = _captured_tile(norm, cell, dx + jx, dy + jy)
            if cap is None:
                continue
            scores[cell] = np.maximum(scores[cell], bank.scores(cap)[:BLANK])

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
    if not assigned:
        return None
    return board, float(min(assigned)), float(np.mean(assigned))


def read_slide(frame, assets: ClueAssets, hint: Optional[float] = None,
               debug_dir: Optional[object] = None) -> Optional[SlideBoard]:
    frame = detection.to_array(frame)
    m = detection.calibrate_scale(frame, assets.needle("slide"), hint=hint)
    if not m.ok:
        return None
    logger.info("Slide interface at (%d, %d), scale %.3f, zncc %.3f", m.x, m.y, m.scale, m.zncc)

    # rank themes at the nominal board origin, refining scale at the same time
    best = None  # (score, scale, norm, ranking)
    for s in (m.scale * 0.95, m.scale, m.scale * 1.05):
        norm = _normalize_board(frame, m, s)
        if norm is None:
            continue
        ranking = sorted(
            ((_probe_score(assets, norm, theme), theme) for theme in SLIDE_THEMES),
            reverse=True,
        )
        if best is None or ranking[0][0] > best[0]:
            best = (ranking[0][0], s, norm, ranking)
    if best is None:
        logger.info("Slide board region out of frame")
        return None
    _, scale, norm, ranking = best
    logger.info("Slide probe: %s", ", ".join(f"{t} {sc:.2f}" for sc, t in ranking[:3]))

    # try the nominal origin first; when that read fails the gates, sweep
    # origins around it (interface revisions move the board) with the
    # top-ranked themes and take the best passing read
    def attempt(dx, dy, theme):
        read = _classify_board(assets, norm, theme, dx, dy)
        if read is None:
            return None
        board, min_sc, conf = read
        logger.info(
            "Slide read theme %s origin (%+d,%+d): min %.2f mean %.2f",
            theme, dx, dy, min_sc, conf,
        )
        if min_sc >= 0.25 and conf >= 0.45:
            return board, theme, dx, dy, conf
        return None

    result = attempt(0, 0, ranking[0][1])
    if result is None and m.zncc < 0.9:
        # the anchor sprite is just a close button; a weak match is almost
        # certainly some other interface's X, not worth the origin sweep
        logger.info("Slide anchor weak (zncc %.2f) and nominal read failed; giving up", m.zncc)
        return None
    if result is None:
        sweep = range(-ORIGIN_SWEEP, ORIGIN_SWEEP + 1, 7)
        candidates = []
        for probe_sc, theme in ranking[:5]:
            top = max(
                ((_probe_score(assets, norm, theme, dx, dy), dx, dy)
                 for dx in sweep for dy in sweep),
                key=lambda t: t[0],
            )
            if top[0] > 0.35:
                candidates.append((top[0], top[1], top[2], theme))
        candidates.sort(reverse=True)
        for _, dx, dy, theme in candidates:
            result = attempt(dx, dy, theme)
            if result is not None:
                break

    if result is None:
        logger.info("Slide read rejected for all candidates")
        if debug_dir is not None:
            try:
                path = str(debug_dir) + "/debug_slide_reject.png"
                Image.fromarray(norm[:, :, :3].astype(np.uint8)).save(path)
                half = Image.fromarray(frame[:, :, :3].astype(np.uint8)).reduce(2)
                half.convert("RGB").save(str(debug_dir) + "/debug_slide_frame.jpg", quality=70)
                logger.info("Saved rejected slide board crop to %s", path)
            except Exception:
                pass
        return None

    board, theme, dx, dy, confidence = result
    bx = int(round(m.x + (NEEDLE_TO_BOARD[0] + dx) * scale))
    by = int(round(m.y + (NEEDLE_TO_BOARD[1] + dy) * scale))
    size = int(round(BOARD_PX * scale))
    return SlideBoard(
        theme=theme,
        board=board,
        scale=scale,
        board_rect=(bx, by, size, size),
        confidence=confidence,
        needle=m,
    )
