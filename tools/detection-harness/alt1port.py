"""Faithful Python ports of Alt1's detection primitives.

Sources: skillbert/alt1 src/base/imagedetect.ts (findSubbuffer/simpleCompare)
and src/ocr/index.ts (readChar/findChar/readLine/readSmallCapsBackwards).
Semantics are kept 1:1 with the JS (thresholds 30/400, checklist prefilter,
canblend scoring, 60% pixel rule) so results match what the real clue solver
would compute on the same pixels.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

MAXDIF = 30  # findSubbuffer per-pixel color budget
MAXPENALTY = 400  # readChar total penalty budget


# ---------------------------------------------------------------- images

def load_rgba(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA")).astype(np.int32)


def app_view(native: np.ndarray) -> np.ndarray:
    """Replicate grab_game: NEAREST downscale of the 2x native frame to 1x."""
    im = Image.fromarray(native.astype(np.uint8))
    im = im.resize((im.width // 2, im.height // 2), Image.NEAREST)
    return np.asarray(im).astype(np.int32)


# ---------------------------------------------------------------- findSubimage

def find_subbuffer(haystack: np.ndarray, needle: np.ndarray,
                   sx=0, sy=0, sw=None, sh=None, max_results=50):
    """Port of ImageDetect.findSubbuffer. Returns list of (x, y) exact hits."""
    hh, hw = haystack.shape[:2]
    nh, nw = needle.shape[:2]
    sw = hw if sw is None else sw
    sh = hh if sh is None else sh

    # checklist: first 10 fully-opaque needle pixels (row-major, y outer)
    ys, xs = np.nonzero(needle[:, :, 3] == 255)
    order = np.lexsort((xs, ys))
    check = list(zip(xs[order][:10], ys[order][:10]))
    if not check:
        return []

    x0, y0 = sx, sy
    x1 = min(sx + sw, hw) - nw
    y1 = min(sy + sh, hh) - nh
    if x1 < x0 or y1 < y0:
        return []

    region_h = y1 - y0 + 1
    region_w = x1 - x0 + 1
    mask = np.ones((region_h, region_w), dtype=bool)
    hs = haystack[:, :, :3]
    for px, py in check:
        window = hs[y0 + py:y0 + py + region_h, x0 + px:x0 + px + region_w]
        d = np.abs(window - needle[py, px, :3]).sum(axis=2)
        mask &= d <= MAXDIF
        if not mask.any():
            return []

    alpha = needle[:, :, 3:4] / 255.0
    nrgb = needle[:, :, :3]
    hits = []
    for y, x in np.argwhere(mask):
        fx, fy = x0 + x, y0 + y
        crop = hs[fy:fy + nh, fx:fx + nw]
        d = np.abs(crop - nrgb).sum(axis=2) * alpha[:, :, 0]
        if (d <= MAXDIF).all():  # simpleCompare acceptance: no pixel over budget
            hits.append((int(fx), int(fy)))
            if len(hits) > max_results:
                return hits
    return hits


def rmse_at(haystack: np.ndarray, needle: np.ndarray, x: int, y: int) -> float:
    """Port of simpleCompareRMSE (alpha-weighted per-pixel RMSE)."""
    nh, nw = needle.shape[:2]
    crop = haystack[y:y + nh, x:x + nw, :3]
    d = np.abs(crop - needle[:, :, :3]).sum(axis=2).astype(np.float64)
    w = needle[:, :, 3] / 255.0
    return math.sqrt((d * d * w).sum() / w.sum())


# ---------------------------------------------------------------- OCR

@dataclass
class Glyph:
    chr: str
    width: int
    bonus: float
    secondary: bool
    xy: np.ndarray      # (n,2) pixel offsets
    alpha: np.ndarray   # (n,) 0..255 text-color proportion
    lum: np.ndarray | None  # (n,) 0..255 luminance factor (shadow fonts)


class Font:
    def __init__(self, meta: dict):
        self.height = meta["height"]
        self.width = meta["width"]
        self.spacewidth = meta["spacewidth"]
        self.shadow = meta["shadow"]
        self.basey = meta["basey"]
        self.bonusperpixel = meta.get("bonusperpixel", 0)
        self.maxspaces = meta.get("maxspaces", 1)
        step = 4 if self.shadow else 3
        self.glyphs = []
        for c in meta["chars"]:
            px = np.array(c["pixels"], dtype=np.float64).reshape(-1, step)
            self.glyphs.append(Glyph(
                chr=c["chr"], width=c["width"], bonus=c.get("bonus", 0),
                secondary=c.get("secondary", False),
                xy=px[:, :2].astype(np.int64), alpha=px[:, 2],
                lum=px[:, 3] if self.shadow else None,
            ))

    @classmethod
    def load(cls, path: str) -> "Font":
        return cls(json.load(open(path)))


def _canblend(px: np.ndarray, c1: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Vectorized canblend: px (n,3) observed, c1 (n,3) expected, p (n,) mix."""
    with np.errstate(divide="ignore"):
        m = np.where(p >= 1.0, 50.0, np.minimum(50.0, p / (1.0 - p)))
    r = px + (px - c1) * m[:, None]
    return np.maximum(0, np.maximum(-r, r - 255)).max(axis=1)


def read_char(buf: np.ndarray, font: Font, col, x: int, y: int,
              backwards: bool, allow_secondary: bool):
    """Port of OCR.readChar. Returns dict or None. Also usable for scoring."""
    h, w = buf.shape[:2]
    y -= font.basey
    col = np.array(col, dtype=np.float64)
    best = None
    for g in font.glyphs:
        if g.secondary and not allow_secondary:
            continue
        n_orig = len(g.xy)
        chrx = x - g.width if backwards else x
        gx = chrx + g.xy[:, 0]
        gy = y + g.xy[:, 1]
        # TS checks bounds against x+subx in both directions; keep that, plus
        # clamp real sample coords so numpy can't wrap negative indices
        bx = x + g.xy[:, 0]
        ok = (gy >= 0) & (gy < h) & (bx >= 0) & (bx < w) & (gx >= 0) & (gx < w)
        ok &= np.take(buf[:, :, 3], np.clip(gy, 0, h - 1) * w + np.clip(gx, 0, w - 1)) >= 128
        cnt = int(ok.sum())
        if cnt < n_orig * 0.6 or (n_orig <= 6 and cnt < n_orig):
            continue
        px = buf[gy[ok], gx[ok], :3].astype(np.float64)
        p = g.alpha[ok] / 255.0
        c1 = col[None, :] * (g.lum[ok, None] / 255.0) if font.shadow else np.broadcast_to(col, (cnt, 3))
        pen = _canblend(px, c1, p)
        score = pen.sum()
        if score > MAXPENALTY:
            continue
        sizescore = score - g.bonus + (n_orig - cnt) * font.bonusperpixel
        if best is None or sizescore < best["sizescore"]:
            best = {"chr": g.chr, "glyph": g, "x": x, "y": y + font.basey,
                    "score": score, "sizescore": sizescore}
    return best


def find_char(buf: np.ndarray, font: Font, col, x, y, w, h):
    if x < 0 or y - font.basey < 0:
        return None
    if x + w + font.width > buf.shape[1] or y + h - font.basey + font.height > buf.shape[0]:
        return None
    best, bestchar = 1000, None
    for cx in range(x, x + w):
        for cy in range(y, y + h):
            c = read_char(buf, font, col, cx, cy, False, False)
            if c is not None and c["sizescore"] < best:
                best, bestchar = c["sizescore"], c
    return bestchar


def read_line(buf: np.ndarray, font: Font, col, x: int, y: int,
              forward=True, backward=True):
    """Mono-color port of OCR.readLine (walks both directions from x,y)."""
    text_parts = []

    def walk(dirforward: bool):
        out = []
        dx, tried = 0, 0
        while True:
            c = read_char(buf, font, col, x + dx, y, not dirforward, True)
            if c is None:
                if tried < font.maxspaces:
                    dx += (1 if dirforward else -1) * font.spacewidth
                    tried += 1
                    continue
                break
            out.append((tried, c["chr"], c["glyph"].secondary))
            tried = 0
            dx += (1 if dirforward else -1) * c["glyph"].width
        return out

    fwd = walk(True) if forward else []
    bwd = walk(False) if backward else []
    for spaces, ch, _ in bwd:  # walk order: nearest first, moving left
        text_parts.insert(0, ch + " " * spaces)
    for spaces, ch, _ in fwd:
        text_parts.append(" " * spaces + ch)
    text = "".join(text_parts)
    had_primary = any(not s for _, _, s in fwd + bwd)
    return text if had_primary else ""


def find_read_line(buf: np.ndarray, font: Font, col, x: int, y: int):
    w = font.width + font.spacewidth
    x -= math.ceil(w / 2)
    h = 7
    y -= 1
    c = find_char(buf, font, col, x, y, w, h)
    if c is None:
        return ""
    return read_line(buf, font, col, c["x"], c["y"])


def read_smallcaps_backwards(buf: np.ndarray, font: Font, col, x: int, y: int,
                             w=-1, h=-1):
    if w == -1:
        w = font.width + font.spacewidth
        x -= math.ceil(w / 2)
    if h == -1:
        h = 7
        y -= 1
    matched = None
    for cx in range(x + w - 1, x - 1, -1):
        best, bestchar = 1000, None
        for cy in range(y, y + h):
            c = read_char(buf, font, col, cx, cy, True, False)
            if c is not None and c["sizescore"] < best:
                best, bestchar = c["sizescore"], c
        if bestchar:
            matched = bestchar
            break
    if matched is None:
        return ""
    return read_line(buf, font, col, matched["x"], matched["y"], forward=False, backward=True)
