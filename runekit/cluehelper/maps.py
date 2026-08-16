"""Map (image) clue identification.

The runeapps clue database stores a signature per map clue: block-wise
hue/saturation/relative-luminance means plus edge densities over the scroll
drawing area. The same signature is computed from the scale-normalized
capture and matched by the reference distance function. Ports of the S/T
feature functions from the clue solver bundle.
"""
import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from runekit import detection
from .assets import ClueAssets

logger = logging.getLogger(__name__)

SCROLL_W, SCROLL_H = 496, 293  # modal rect size used by the signature
SIG_RECT = (90, 25, 300, 240)  # sampled area within the scroll
BLOCK = 20


def _hsl(rgb: np.ndarray) -> Tuple[int, int, int]:
    r, g, b = (float(v) / 256.0 for v in rgb)
    mx, mn = max(r, g, b), min(r, g, b)
    chroma = mx - mn
    lum = 0.5 * (mx + mn)
    hue = 0.0
    if chroma != 0:
        if mx == r:
            hue = (6 + (g - b) / chroma) % 6
        if mx == g:
            hue = (b - r) / chroma + 2
        if mx == b:
            hue = (r - g) / chroma + 4
    return (round(hue / 6 * 255), round(255 * chroma), round(255 * lum))


def _block_mean(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    return img[y:y + h, x:x + w, :3].reshape(-1, 3).mean(axis=0)


def _edge_energy(img: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    a = img[y:y + h + 1, x:x + w + 1, :3].astype(np.int64)
    dx = np.abs(a[:h, :w] - a[:h, 1:w + 1]).sum()
    dy = np.abs(a[:h, :w] - a[1:h + 1, :w]).sum()
    return float(dx + dy)


def signature(img: np.ndarray) -> List[int]:
    """Port of the bundle's S() over the scroll drawing area. img must be the
    normalized 1x scroll (SCROLL_W x SCROLL_H)."""
    x0, y0, w, h = SIG_RECT
    overall = _hsl(_block_mean(img, x0, y0, w, h))
    cols = w // BLOCK
    sig = [overall[0], overall[1], overall[2]] + [0] * (5 * cols * (h // BLOCK))
    for c in range(cols):
        bx = x0 + c * BLOCK
        for r in range(h // BLOCK):
            by = y0 + r * BLOCK
            i = 5 * c + r * cols * 5 + 3
            d = _hsl(_block_mean(img, bx, by, BLOCK, BLOCK))
            sig[i + 0] = d[0]
            sig[i + 1] = d[1]
            sig[i + 2] = overall[2] - d[2]
            sig[i + 3] = int(_edge_energy(img, bx + 1, by + 1, BLOCK - 2, BLOCK - 2) / BLOCK / BLOCK)
            sig[i + 4] = int(_edge_energy(img, bx, by, BLOCK, BLOCK) / BLOCK / BLOCK)
    return sig


def sig_distance(a: List[int], b: List[int]) -> float:
    """Port of the bundle's T() distance."""
    n = 0.0
    r = abs(a[0] - b[0])
    n += max(0.0, 5 * (255 - r if r > 128 else r) - 100)
    n += max(0.0, 5 * abs(a[1] - b[1]) - 100)
    for i in range(3, min(len(a), len(b)), 5):
        r = abs(a[i] - b[i])
        o = (255 - r if r > 128 else r) * max(a[i + 1], b[i + 1]) / 255
        o += abs(a[i + 1] - b[i + 1])
        o += 100 * max(0, a[i + 3] - b[i + 4])
        o += 100 * max(0, b[i + 3] - a[i + 4])
        n += o
    return n


@dataclass
class MapMatch:
    entry: dict
    score: float
    margin: float  # distance gap to the runner-up


def find_modal(frame: np.ndarray, assets: ClueAssets, hint: Optional[float] = None):
    """Locate an eoc modal by close button + top-left corner.
    Returns (rect_x, rect_y, scale) in frame coords, or None."""
    exitbtn = detection.to_array(assets.needle("exitbutton"))
    topleft = detection.to_array(assets.needle("topleft"))
    x_match = detection.calibrate_scale(frame, exitbtn, hint=hint)
    if not x_match.ok:
        return None
    s = x_match.scale
    x0 = max(0, int(x_match.x - 900 * s))
    y0 = max(0, int(x_match.y - 60 * s))
    y1 = min(frame.shape[0], int(x_match.y + 60 * s))
    sub = frame[y0:y1, x0:x_match.x]
    corner = detection.find_template(sub, topleft, np.linspace(0.94, 1.06, 7) * s)
    if corner.zncc < 0.5:
        return None
    return (x0 + corner.x + 4 * s, x_match.y + 24 * s, s)


def read_map_clue(frame, dbs: dict, assets: ClueAssets, hint: Optional[float] = None) -> Optional[MapMatch]:
    frame = detection.to_array(frame)
    refs = []
    for e in dbs["clues"]:
        if e.get("type") not in ("img", "emptyimg"):
            continue
        sig = e.get("clue")
        if isinstance(sig, str):
            sig = json.loads(sig)
        if isinstance(sig, list) and len(sig) > 100:
            refs.append((e, sig))
    if not refs:
        return None

    modal = find_modal(frame, assets, hint=hint)
    if modal is None:
        logger.info("Map clue: modal not located")
        return None
    rx, ry, s = modal

    best = None
    for cs in (s * 0.97, s, s * 1.03):
        w = int(round((SCROLL_W + 2) * cs))
        h = int(round((SCROLL_H + 2) * cs))
        x0, y0 = int(round(rx)), int(round(ry))
        if x0 < 0 or y0 < 0 or x0 + w > frame.shape[1] or y0 + h > frame.shape[0]:
            continue
        norm = detection.normalize_region(frame, x0, y0, w, h, cs)
        if norm.shape[0] < SCROLL_H + 1 or norm.shape[1] < SCROLL_W + 1:
            continue
        sig = signature(norm)
        scored = sorted((sig_distance(sig, ref), e) for e, ref in refs)
        score, entry = scored[0]
        margin = scored[1][0] - score if len(scored) > 1 else score
        if best is None or score < best.score:
            best = MapMatch(entry=entry, score=score, margin=margin)

    if best is None:
        return None
    logger.info(
        "Map clue best match clueid %s score %.0f margin %.0f",
        best.entry.get("clueid"), best.score, best.margin,
    )
    # empirical: true matches score ~20k even through scaling, wrong maps 130k+
    if best.score > 60000 or best.margin < best.score:
        return None
    return best
