"""Offline harness: runs the clue solver's exact detection (ported from Alt1)
against captured frames, plus scale-swept scoring to evaluate config-scale
tolerant detection.

Usage:
  python run_harness.py --sanity                 # synthetic round-trip tests
  python run_harness.py --frame path.png         # exact + scale sweep on frame
  python run_harness.py --frame path.png --scale 1.5   # score at fixed scale
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alt1port as a1

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.environ.get("CLUE_ASSETS") or os.path.join(HERE, "assets")

NEEDLES = ["exitbutton", "topleft", "botleft", "homeportbutton", "northimg"]


def load_needle(name):
    return np.asarray(Image.open(os.path.join(ASSETS, f"needle_{name}.png")).convert("RGBA")).astype(np.int32)


def composite(needle, bg):
    a = needle[:, :, 3:4] / 255.0
    return (needle[:, :, :3] * a + bg * (1 - a)).astype(np.int32)


# ---------------------------------------------------------------- sanity

def sanity():
    rng = np.random.default_rng(7)
    ok = True

    # 1. exact subimage finder round trip
    needle = load_needle("exitbutton")
    frame = np.zeros((400, 600, 4), dtype=np.int32)
    frame[:, :, :3] = rng.integers(0, 256, (400, 600, 3))
    frame[:, :, 3] = 255
    px, py = 231, 117
    nh, nw = needle.shape[:2]
    frame[py:py + nh, px:px + nw, :3] = composite(needle, frame[py:py + nh, px:px + nw, :3])
    hits = a1.find_subbuffer(frame, needle)
    print(f"[sanity] find_subbuffer: hits={hits} expected=[({px}, {py})]")
    ok &= hits == [(px, py)]

    # 2. OCR round trip: render text from the font's own glyph data, read back
    for fontname, text, col, y in [
        ("font_chat12", "Search the crates near the", (84, 72, 56), 40),
        ("font_allcaps9", "MYSTERIOUS CLUE SCROLL", (255, 203, 5), 80),
    ]:
        font = a1.Font.load(os.path.join(ASSETS, f"{fontname}.fontmeta.json"))
        buf = np.zeros((120, 400, 4), dtype=np.int32)
        buf[:, :, :3] = (205, 175, 135)  # parchment-ish bg
        buf[:, :, 3] = 255
        glyphs = {g.chr: g for g in font.glyphs}
        x = 30
        x_mid = None
        for ch in text:
            if ch == " ":
                x += font.spacewidth
                continue
            g = glyphs[ch]
            for (gx, gy), al, lum in zip(g.xy, g.alpha, g.lum if font.shadow else np.full(len(g.xy), 255.0)):
                p = al / 255.0
                c = np.array(col) * (lum / 255.0 if font.shadow else 1.0)
                fx, fy = x + int(gx), y - font.basey + int(gy)
                buf[fy, fx, :3] = (c * p + buf[fy, fx, :3] * (1 - p)).astype(np.int32)
            x += g.width
            if x_mid is None:
                x_mid = x  # somewhere inside the text
        got = a1.find_read_line(buf, font, col, x_mid + 20, y)
        match = got.strip() == text
        print(f"[sanity] {fontname} round trip: {got.strip()!r} match={match}")
        ok &= match

    print(f"[sanity] {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return ok


# ---------------------------------------------------------------- scaled scoring

def scale_needle(needle, s, resample=Image.BILINEAR):
    im = Image.fromarray(needle.astype(np.uint8))
    im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), resample)
    return np.asarray(im).astype(np.int32)


def masked_min_rmse(frame, needle):
    """Best (lowest) alpha-masked per-pixel RMSE of needle anywhere in frame."""
    import cv2
    f = frame[:, :, :3].astype(np.float32)
    t = needle[:, :, :3].astype(np.float32)
    m = np.repeat((needle[:, :, 3:4] / 255.0).astype(np.float32), 3, axis=2)
    res = cv2.matchTemplate(f, t, cv2.TM_SQDIFF, mask=m)
    minv, _, minloc, _ = cv2.minMaxLoc(res)
    denom = float(m.sum())
    rmse = float(np.sqrt(max(0.0, minv) / denom))
    return rmse, minloc


def sweep(frame, needle, scales, resample=Image.BILINEAR):
    rows = []
    for s in scales:
        sc = scale_needle(needle, s, resample)
        if sc.shape[0] >= frame.shape[0] or sc.shape[1] >= frame.shape[1]:
            continue
        rmse, loc = masked_min_rmse(frame, sc)
        rows.append({"scale": round(float(s), 3), "rmse": round(rmse, 2), "x": loc[0], "y": loc[1]})
    return sorted(rows, key=lambda r: r["rmse"])


# ---------------------------------------------------------------- frame run

def run_frame(path, fixed_scale=None):
    print(f"=== frame: {path}")
    native = a1.load_rgba(path)
    views = {"appview(1x-decimated)": a1.app_view(native), "native(2x)": native}

    for view_name, frame in views.items():
        print(f"-- {view_name}: {frame.shape[1]}x{frame.shape[0]}")
        t0 = time.time()
        for name in NEEDLES:
            hits = a1.find_subbuffer(frame, load_needle(name))
            print(f"   exact {name:15s}: {len(hits)} hits {hits[:5]}")
        print(f"   (exact search took {time.time() - t0:.1f}s)")

    frame = views["appview(1x-decimated)"]
    if fixed_scale:
        print(f"-- fixed scale {fixed_scale} (config mode), appview:")
        for name in NEEDLES:
            sc = scale_needle(load_needle(name), fixed_scale)
            rmse, loc = masked_min_rmse(frame, sc)
            print(f"   {name:15s}: rmse={rmse:6.2f} at {loc}")
    else:
        print("-- scale sweep (exitbutton, appview, bilinear-scaled needle):")
        rows = sweep(frame, load_needle("exitbutton"), np.arange(0.75, 3.01, 0.125))
        for r in rows[:6]:
            print(f"   scale={r['scale']:<6} rmse={r['rmse']:<8} at ({r['x']},{r['y']})")
        best = rows[0]
        print(f"   BEST: scale={best['scale']} rmse={best['rmse']}")
        json.dump(rows, open(os.path.join(HERE, "sweep_exitbutton.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--frame")
    ap.add_argument("--scale", type=float)
    args = ap.parse_args()
    if args.sanity:
        sys.exit(0 if sanity() else 1)
    if args.frame:
        run_frame(args.frame, args.scale)
