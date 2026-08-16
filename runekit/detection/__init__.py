"""Scale-tolerant template matching for scaled/blurry game interfaces.

The macOS client renders at Retina resolution with in-game interface scaling
on top, so captured UI is never pixel-identical to 1x reference sprites.
Matching here is confidence-scored: templates are degraded the way the
pipeline degrades the real UI, located with masked SQDIFF, scored with masked
zero-mean normalized cross-correlation, and gated on the margin over the
frame's background score distribution.
"""
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from PIL import Image


def to_array(image) -> np.ndarray:
    """Any frame/needle input to an int32 RGBA array."""
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGBA"))
    image = image.astype(np.int32)
    if image.shape[2] == 3:
        out = np.zeros((*image.shape[:2], 4), dtype=np.int32)
        out[:, :, :3] = image
        out[:, :, 3] = 255
        return out
    return image


def zncc(frame: np.ndarray, template: np.ndarray, x: int, y: int) -> float:
    """Masked zero-mean normalized cross-correlation of template at (x, y)."""
    th, tw = template.shape[:2]
    crop = frame[y:y + th, x:x + tw, :3].astype(np.float64)
    if crop.shape[:2] != (th, tw):
        return -1.0
    t = template[:, :, :3].astype(np.float64)
    w = (template[:, :, 3:4] / 255.0).astype(np.float64)
    n = w.sum() * 3
    if n < 1:
        return -1.0
    cm = (crop * w).sum() / n
    tm = (t * w).sum() / n
    cz, tz = (crop - cm) * w, (t - tm) * w
    denom = math.sqrt((cz * cz).sum() * (tz * tz).sum())
    return float((cz * tz).sum() / denom) if denom else -1.0


def degrade_template(template: np.ndarray, scale: float) -> np.ndarray:
    """Degrade a 1x template the way the capture pipeline degrades game UI:
    bilinear upscale to 2*scale (game render), NEAREST decimate by 2 (capture)."""
    im = Image.fromarray(template.astype(np.uint8))
    big = im.resize(
        (max(2, round(im.width * 2 * scale)), max(2, round(im.height * 2 * scale))),
        Image.BILINEAR,
    )
    dec = big.resize((big.width // 2, big.height // 2), Image.NEAREST)
    return np.asarray(dec).astype(np.int32)


def _masked_sqdiff(frame32: np.ndarray, template: np.ndarray):
    import cv2

    t = template[:, :, :3].astype(np.float32)
    m = np.repeat((template[:, :, 3:4] / 255.0).astype(np.float32), 3, axis=2)
    return cv2.matchTemplate(frame32, t, cv2.TM_SQDIFF, mask=m)


def locate(frame: np.ndarray, template: np.ndarray):
    """Best position of template in frame by masked SQDIFF. Returns (x, y).

    Masked matching has no FFT fast path, so on large frames the search runs
    on a half-resolution pyramid and the best peaks are refined at full
    resolution in small windows."""
    import cv2

    f32 = frame[:, :, :3].astype(np.float32)
    th, tw = template.shape[:2]

    small_enough = frame.shape[0] * frame.shape[1] <= 1_500_000
    if small_enough or min(th, tw) < 8:
        res = _masked_sqdiff(f32, template)
        _, _, loc, _ = cv2.minMaxLoc(res)
        return loc

    im = Image.fromarray(template.astype(np.uint8))
    t2 = np.asarray(
        im.resize((max(2, tw // 2), max(2, th // 2)), Image.BILINEAR)
    ).astype(np.int32)
    f2 = cv2.resize(f32, (f32.shape[1] // 2, f32.shape[0] // 2), interpolation=cv2.INTER_AREA)
    res = _masked_sqdiff(f2, t2)

    # take several coarse peaks with suppression, refine each at full res
    best = None
    r = res.copy()
    for _ in range(5):
        _, _, loc, _ = cv2.minMaxLoc(r)
        px, py = loc[0] * 2, loc[1] * 2
        y0 = max(0, py - 4)
        x0 = max(0, px - 4)
        y1 = min(frame.shape[0], py + th + 4)
        x1 = min(frame.shape[1], px + tw + 4)
        if y1 - y0 > th and x1 - x0 > tw:
            sub = _masked_sqdiff(f32[y0:y1, x0:x1], template)
            val, _, sloc, _ = cv2.minMaxLoc(sub)
            cand = (val, (x0 + sloc[0], y0 + sloc[1]))
            if best is None or cand[0] < best[0]:
                best = cand
        sy0 = max(0, loc[1] - t2.shape[0])
        sx0 = max(0, loc[0] - t2.shape[1])
        r[sy0:loc[1] + t2.shape[0], sx0:loc[0] + t2.shape[1]] = np.inf
    if best is None:
        _, _, loc, _ = cv2.minMaxLoc(res)
        return (loc[0] * 2, loc[1] * 2)
    return best[1]


@dataclass
class Match:
    scale: float
    x: int
    y: int
    zncc: float
    margin: float  # zncc minus the best background score

    @property
    def ok(self) -> bool:
        return self.zncc >= 0.75 and self.margin >= 0.12


def find_template(
    frame,
    template,
    scales: Sequence[float],
    rng: Optional[np.random.Generator] = None,
) -> Match:
    """Search template over candidate scales; return the best hit with a
    confidence margin over random background locations."""
    frame = to_array(frame)
    template = to_array(template)
    rng = rng or np.random.default_rng(3)

    best = None
    best_t = None
    for s in scales:
        t = degrade_template(template, s)
        if t.shape[0] >= frame.shape[0] or t.shape[1] >= frame.shape[1]:
            continue
        x, y = locate(frame, t)
        score = zncc(frame, t, x, y)
        if best is None or score > best.zncc:
            best = Match(scale=float(s), x=int(x), y=int(y), zncc=score, margin=0.0)
            best_t = t
    if best is None:
        return Match(scale=1.0, x=0, y=0, zncc=-1.0, margin=-1.0)

    th, tw = best_t.shape[:2]
    bg = max(
        zncc(
            frame,
            best_t,
            int(rng.integers(0, frame.shape[1] - tw)),
            int(rng.integers(0, frame.shape[0] - th)),
        )
        for _ in range(60)
    )
    best.margin = best.zncc - bg
    return best


def calibrate_scale(frame, needle, coarse=(1.0, 3.0), fine_step=0.03125,
                    hint: Optional[float] = None) -> Match:
    """Two-pass scale calibration against a known needle. A hint (the scale
    found on a previous solve) is tried as a narrow fast path first, falling
    back to the full sweep when it no longer matches."""
    frame = to_array(frame)
    needle = to_array(needle)
    if hint:
        m = find_template(frame, needle, np.arange(hint * 0.94, hint * 1.06, fine_step))
        if m.ok:
            return m
    first = find_template(frame, needle, np.arange(coarse[0], coarse[1] + 1e-9, 0.125))
    if first.zncc < 0.5:
        return first
    lo, hi = first.scale - 0.125, first.scale + 0.125
    return find_template(frame, needle, np.arange(lo, hi + 1e-9, fine_step))


class TemplateBank:
    """Same-size RGB templates prepared for batch ZNCC scoring: one matrix
    product scores a crop against every template at once."""

    def __init__(self, arrays: Sequence[np.ndarray]):
        stack = np.stack([a[:, :, :3] for a in arrays]).astype(np.float64)
        self.count = stack.shape[0]
        self.shape = stack.shape[1:3]
        flat = stack.reshape(self.count, -1)
        z = flat - flat.mean(axis=1, keepdims=True)
        norms = np.sqrt((z * z).sum(axis=1))
        self.z = z
        self.norms = np.maximum(norms, 1e-9)

    def scores(self, crop: np.ndarray) -> np.ndarray:
        c = crop[:, :, :3].astype(np.float64).reshape(-1)
        c = c - c.mean()
        cn = math.sqrt((c * c).sum())
        if cn < 1e-9:
            return np.full(self.count, -1.0)
        return (self.z @ c) / (self.norms * cn)


def normalize_region(frame, x: int, y: int, w: int, h: int, scale: float) -> np.ndarray:
    """Crop a region from the served frame and resample it to 1x reference
    geometry with a good filter."""
    frame = to_array(frame)
    crop = frame[y:y + h, x:x + w, :3].astype(np.uint8)
    im = Image.fromarray(crop)
    im = im.resize(
        (max(1, round(im.width / scale)), max(1, round(im.height / scale))),
        Image.LANCZOS,
    )
    return to_array(np.asarray(im))
