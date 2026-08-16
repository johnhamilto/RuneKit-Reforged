"""Confidence-scored detection and OCR on scaled/blurry interfaces.

Compares three tiers on the same pixels:
  1. Alt1 exact matching (binary, ported in alt1port)  -> fails when scaled
  2. Scale-normalized soft matching with confidence scores
  3. Closed-world lookup: fuzzy match OCR output against clues.json
Optionally benchmarks Apple Vision OCR if pyobjc-framework-Vision is present.
"""
import difflib
import json
import math
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.environ.get("CLUE_ASSETS") or os.path.join(HERE, "assets")
sys.path.insert(0, HERE)
import alt1port as a1


# ---------------------------------------------------------------- detection confidence

def zncc(frame, template, mask, x, y):
    """Masked zero-mean normalized cross-correlation at one location."""
    th, tw = template.shape[:2]
    crop = frame[y:y + th, x:x + tw, :3].astype(np.float64)
    t = template[:, :, :3].astype(np.float64)
    w = (mask / 255.0)[:, :, None]
    n = w.sum() * 3
    cm = (crop * w).sum() / n
    tm = (t * w).sum() / n
    cz, tz = (crop - cm) * w, (t - tm) * w
    denom = math.sqrt((cz * cz).sum() * (tz * tz).sum())
    return float((cz * tz).sum() / denom) if denom else 0.0


def pipeline_template(needle, s):
    """Degrade the needle the way the real pipeline degrades the game UI:
    bilinear upscale to 2*s (game render at Retina * interface scale), then
    NEAREST decimate by 2 (grab_game)."""
    im = Image.fromarray(needle.astype(np.uint8))
    big = im.resize((max(2, round(im.width * 2 * s)), max(2, round(im.height * 2 * s))), Image.BILINEAR)
    dec = big.resize((big.width // 2, big.height // 2), Image.NEAREST)
    return np.asarray(dec).astype(np.int32)


def detect_confident(frame, needle, scales, rng=None):
    """Sweep scales, score best location per scale with masked ZNCC, and return
    best hit with a z-score against the frame's background score distribution."""
    import cv2
    rng = rng or np.random.default_rng(11)
    f32 = frame[:, :, :3].astype(np.float32)
    best = None
    for s in scales:
        t = pipeline_template(needle, s)
        m3 = np.repeat((t[:, :, 3:4] / 255.0).astype(np.float32), 3, axis=2)
        res = cv2.matchTemplate(f32, t[:, :, :3].astype(np.float32), cv2.TM_SQDIFF, mask=m3)
        _, _, loc, _ = cv2.minMaxLoc(res)
        score = zncc(frame, t, t[:, :, 3], loc[0], loc[1])
        if best is None or score > best["zncc"]:
            best = {"scale": float(s), "x": loc[0], "y": loc[1], "zncc": score, "template": t}
    # background distribution: ZNCC of the winning template at random spots
    t = best["template"]
    th, tw = t.shape[:2]
    bg = [zncc(frame, t, t[:, :, 3],
               int(rng.integers(0, frame.shape[1] - tw)),
               int(rng.integers(0, frame.shape[0] - th)))
          for _ in range(200)]
    mu, sd = float(np.mean(bg)), float(np.std(bg))
    best["bg_mean"] = mu
    best["bg_std"] = sd
    best["bg_max"] = float(np.max(bg))
    best["z"] = (best["zncc"] - mu) / sd if sd else 0.0
    best["margin"] = best["zncc"] - best["bg_max"]
    del best["template"]
    return best


# ---------------------------------------------------------------- soft OCR

class GlyphTemplates:
    """Renders every glyph the way it appears in game (text color + alpha over
    a background estimate, shadow luminance applied) and scores observed crops
    with zero-mean normalized cross-correlation over the FULL glyph box, so ink
    where the template has none counts against the match."""

    def __init__(self, font, col, bg):
        self.font = font
        self.h = font.height
        col = np.asarray(col, dtype=np.float64)
        bg = np.asarray(bg, dtype=np.float64)
        self.templates = []
        for g in font.glyphs:
            t = np.zeros((self.h, max(1, g.width), 3), dtype=np.float64)
            t[:, :] = bg
            lums = g.lum if font.shadow else np.full(len(g.xy), 255.0)
            for (gx, gy), al, lum in zip(g.xy, g.alpha, lums):
                if 0 <= gy < self.h and 0 <= gx < t.shape[1]:
                    p = al / 255.0
                    ink = col * (lum / 255.0 if font.shadow else 1.0)
                    t[int(gy), int(gx)] = ink * p + t[int(gy), int(gx)] * (1 - p)
            tz = t - t.mean()
            norm = math.sqrt((tz * tz).sum())
            self.templates.append((g, tz, norm))

    def score_at(self, buf, x, y, at_end=False):
        """Rank glyphs by ZNCC. x is the glyph's left edge, or its right edge
        when at_end=True (for backward reading). Sorted best-first."""
        h, w = buf.shape[:2]
        top = y - self.font.basey
        rows = []
        for g, tz, tn in self.templates:
            gw = tz.shape[1]
            gx = x - gw if at_end else x
            if tn < 1e-6 or top < 0 or top + self.h > h or gx < 0 or gx + gw > w:
                continue
            crop = buf[top:top + self.h, gx:gx + gw, :3].astype(np.float64)
            cz = crop - crop.mean()
            cn = math.sqrt((cz * cz).sum())
            if cn < 1e-6:
                continue
            rows.append({"chr": g.chr, "width": g.width, "secondary": g.secondary,
                         "zncc": float((cz * tz).sum() / (cn * tn))})
        return sorted(rows, key=lambda r: -r["zncc"])


def softmax_probs(rows, temperature=0.06):
    z = np.array([r["zncc"] for r in rows])
    p = np.exp((z - z.max()) / temperature)
    p /= p.sum()
    for r, pi in zip(rows, p):
        r["p"] = float(pi)
    return rows


def _flat(buf, tmpl, x, y, width, std_thresh=6.0):
    top = y - tmpl.font.basey
    crop = buf[top:top + tmpl.font.height, max(0, x):x + width, :3]
    return crop.size == 0 or float(crop.std()) < std_thresh


def _best_step(buf, tmpl, x, y, at_end, allow_secondary):
    best = None
    for jx in (0, -1, 1):
        for jy in (0, -1, 1):
            cands = tmpl.score_at(buf, x + jx, y + jy, at_end=at_end)
            cands = [c for c in cands if not c["secondary"] or allow_secondary]
            if cands and (best is None or cands[0]["zncc"] > best[0][0]["zncc"]):
                best = (softmax_probs(cands), jx)
    return best


def read_line_soft(buf, tmpl: GlyphTemplates, x, y, max_chars=70,
                   min_zncc=0.45, min_zncc_secondary=0.62):
    """Greedy soft read: forward from the anchor, then a backward extension.
    Flat (background) regions become spaces; three consecutive end the line.
    Returns dict with text, per-char detail, mean softmax prob, mean zncc."""
    font = tmpl.font

    def gate(top):
        return top["zncc"] >= (min_zncc_secondary if top["secondary"] else min_zncc)

    def walk(startx, at_end):
        out, flats = [], 0
        dx = 0
        step = -1 if at_end else 1
        while len(out) < max_chars and flats < 3:
            cx = startx + dx
            probe = cx - font.spacewidth if at_end else cx
            if _flat(buf, tmpl, probe, y, font.spacewidth):
                out.append((" ", 1.0, 1.0, None))
                flats += 1
                dx += step * font.spacewidth
                continue
            best = _best_step(buf, tmpl, cx, y, at_end, allow_secondary=bool(out))
            if best is None or not gate(best[0][0]):
                out.append((" ", 0.5, 0.0, None))
                flats += 1
                dx += step * font.spacewidth
                continue
            cands, jx = best
            top = cands[0]
            out.append((top["chr"], top["p"], top["zncc"],
                        cands[1]["chr"] if len(cands) > 1 else None))
            dx += jx + step * top["width"]
            flats = 0
        while out and out[-1][0] == " ":
            out.pop()
        return out

    fwd = walk(x, at_end=False)
    bwd = walk(x, at_end=True)
    chars = list(reversed(bwd)) + fwd
    while chars and chars[0][0] == " ":
        chars.pop(0)
    text = "".join(c for c, _, _, _ in chars)
    real = [(p, z) for c, p, z, _ in chars if c != " "]
    mean_p = float(np.mean([p for p, _ in real])) if real else 0.0
    mean_z = float(np.mean([z for _, z in real])) if real else 0.0
    return {"text": text, "chars": chars, "mean_p": mean_p, "mean_zncc": mean_z}


def normalize_region(frame, x, y, w, h, scale):
    """Crop a region from the served frame and resample it back to 1x
    reference geometry (the config-scale step)."""
    crop = frame[y:y + h, x:x + w, :3].astype(np.uint8)
    im = Image.fromarray(crop)
    im = im.resize((max(1, round(im.width / scale)), max(1, round(im.height / scale))), Image.LANCZOS)
    arr = np.asarray(im).astype(np.int32)
    out = np.zeros((*arr.shape[:2], 4), dtype=np.int32)
    out[:, :, :3] = arr
    out[:, :, 3] = 255
    return out


# ---------------------------------------------------------------- closed-world lookup

def load_clues():
    return json.load(open(os.path.join(ASSETS, "clues.json")))


def fuzzy_clue_match(text, clues, top=3):
    t = text.lower().strip()
    scored = []
    for c in clues:
        clue = c.get("clue") or ""
        if isinstance(clue, list):
            clue = " ".join(str(s) for s in clue)
        if not clue:
            continue
        r = difflib.SequenceMatcher(None, t, clue.lower()).ratio()
        scored.append((r, c))
    scored.sort(key=lambda x: -x[0])
    return scored[:top]


# ---------------------------------------------------------------- Apple Vision

def vision_ocr(img_arr):
    try:
        import Vision
        import Quartz
    except ImportError:
        return None
    from io import BytesIO
    buf = BytesIO()
    Image.fromarray(img_arr[:, :, :3].astype(np.uint8)).save(buf, format="PNG")
    data = Quartz.CFDataCreate(None, buf.getvalue(), len(buf.getvalue()))
    src = Quartz.CGImageSourceCreateWithData(data, None)
    cgimg = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cgimg, None)
    ok, err = handler.performRequests_error_([req], None)
    if not ok:
        return None
    out = []
    for obs in req.results() or []:
        cand = obs.topCandidates_(1)[0]
        out.append((str(cand.string()), float(cand.confidence())))
    return out
