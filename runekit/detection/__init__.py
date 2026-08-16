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


def locate(frame: np.ndarray, template: np.ndarray):
    """Best position of template in frame by masked SQDIFF. Returns (x, y)."""
    import cv2

    f = frame[:, :, :3].astype(np.float32)
    t = template[:, :, :3].astype(np.float32)
    m = np.repeat((template[:, :, 3:4] / 255.0).astype(np.float32), 3, axis=2)
    res = cv2.matchTemplate(f, t, cv2.TM_SQDIFF, mask=m)
    _, _, loc, _ = cv2.minMaxLoc(res)
    return loc


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
        for _ in range(120)
    )
    best.margin = best.zncc - bg
    return best


def calibrate_scale(frame, needle, coarse=(1.0, 3.0), fine_step=0.03125) -> Match:
    """Two-pass scale calibration against a known needle."""
    frame = to_array(frame)
    needle = to_array(needle)
    first = find_template(frame, needle, np.arange(coarse[0], coarse[1] + 1e-9, 0.125))
    if first.zncc < 0.5:
        return first
    lo, hi = first.scale - 0.125, first.scale + 0.125
    return find_template(frame, needle, np.arange(lo, hi + 1e-9, fine_step))


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
