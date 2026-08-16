"""Experiments: binary Alt1 matching vs confidence-scored detection/OCR.

Exp 1: detection confidence (ZNCC + z-score) on positive/negative real frames
Exp 2: simulated clue text at the user's measured conditions (2x retina,
       ~1.625 interface scale, NEAREST decimation) through 4 readers
Exp 3: real captured chat text through the same readers
"""
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.environ.get("RUNEKIT_FRAMES") or os.path.join(HERE, "frames")
sys.path.insert(0, HERE)
import alt1port as a1
import confidence as cf

SCALE = 1.625  # measured from the harness sweep


def render_text(font, text, col, bg, pad=30):
    glyphs = {g.chr: g for g in font.glyphs}
    width = pad * 2 + sum(glyphs[c].width if c != " " else font.spacewidth for c in text)
    buf = np.zeros((font.height + pad * 2, width, 4), dtype=np.int32)
    buf[:, :, :3] = bg
    buf[:, :, 3] = 255
    x, y = pad, pad + font.basey
    startx = x
    for ch in text:
        if ch == " ":
            x += font.spacewidth
            continue
        g = glyphs[ch]
        lums = g.lum if font.shadow else np.full(len(g.xy), 255.0)
        for (gx, gy), al, lum in zip(g.xy, g.alpha, lums):
            p = al / 255.0
            c = np.array(col) * (lum / 255.0 if font.shadow else 1.0)
            fy, fx = y - font.basey + int(gy), x + int(gx)
            buf[fy, fx, :3] = (c * p + buf[fy, fx, :3] * (1 - p)).astype(np.int32)
        x += g.width
    return buf, startx, y


def degrade(buf, s=SCALE):
    """Game-render + capture pipeline: bilinear x(2s), NEAREST /2."""
    im = Image.fromarray(buf[:, :, :3].astype(np.uint8))
    big = im.resize((round(im.width * 2 * s), round(im.height * 2 * s)), Image.BILINEAR)
    dec = big.resize((big.width // 2, big.height // 2), Image.NEAREST)
    arr = np.asarray(dec).astype(np.int32)
    out = np.zeros((*arr.shape[:2], 4), dtype=np.int32)
    out[:, :, :3] = arr
    out[:, :, 3] = 255
    return out


def anchor_search(buf, tmpl, x0, y0, rx=5, ry=4):
    best = (-1.0, x0, y0)
    for x in range(x0 - rx, x0 + rx + 1):
        for y in range(y0 - ry, y0 + ry + 1):
            c = [r for r in tmpl.score_at(buf, x, y) if not r["secondary"]]
            if c and c[0]["zncc"] > best[0]:
                best = (c[0]["zncc"], x, y)
    return best[1], best[2]


def show_read(tag, res):
    if isinstance(res, str):
        print(f"   {tag:34s} -> {res!r}")
        return
    print(f"   {tag:34s} -> {res['text']!r}  (mean p {res['mean_p']:.2f}, mean zncc {res['mean_zncc']:.2f})")
    low = [f"{c}:{p:.2f}{'/' + a if a else ''}" for c, p, z, a in res["chars"] if c != ' ' and p < 0.6]
    if low:
        print(f"     low-confidence chars: {', '.join(low[:8])}")


def main():
    clues = cf.load_clues()

    print("== Exp 1: detection confidence (exitbutton, pipeline template) ==")
    for name, path, expected in [("positive (loot X on screen)", "smoke_window_prev.png", True),
                                 ("negative (no modal)", "clue_native.png", False)]:
        frame = a1.app_view(a1.load_rgba(os.path.join(SCRATCH, path)))
        needle = np.asarray(Image.open(os.path.join(cf.ASSETS, "needle_exitbutton.png")).convert("RGBA")).astype(np.int32)
        r = cf.detect_confident(frame, needle, np.arange(1.25, 2.26, 0.125))
        print(f"   {name:28s}: zncc={r['zncc']:.3f} scale={r['scale']:.3f} at ({r['x']},{r['y']})"
              f"  bgmax={r['bg_max']:.3f}  margin={r['margin']:.3f}  z={r['z']:.1f}")

    print("\n== Exp 2: simulated clue text at measured conditions ==")
    truth = "Search the crates near the Lumbridge Market."
    font12 = a1.Font.load(os.path.join(cf.ASSETS, "font_chat12.fontmeta.json"))
    clean, sx, sy = render_text(font12, truth, (84, 72, 56), (205, 175, 135))
    served = degrade(clean)
    Image.fromarray(served[:, :, :3].astype(np.uint8)).save(os.path.join(SCRATCH, "exp2_served.png"))
    mid = (sx + 60, sy)

    t = a1.find_read_line(served, font12, (84, 72, 56), int(mid[0] * SCALE), int(mid[1] * SCALE))
    show_read("A alt1-exact on served frame", t)

    norm = cf.normalize_region(served, 0, 0, served.shape[1], served.shape[0], SCALE)
    Image.fromarray(norm[:, :, :3].astype(np.uint8)).save(os.path.join(SCRATCH, "exp2_normalized.png"))
    t = a1.find_read_line(norm, font12, (84, 72, 56), *mid)
    show_read("B alt1-exact on normalized", t)

    tmpl = cf.GlyphTemplates(font12, (84, 72, 56), (205, 175, 135))
    ax, ay = anchor_search(norm, tmpl, sx, sy)
    res = cf.read_line_soft(norm, tmpl, ax, ay)
    show_read("C soft OCR on normalized", res)
    for r, c in cf.fuzzy_clue_match(res["text"], clues):
        print(f"     clue match {r:.3f}: {str(c['clue'])[:60]!r}")

    v = cf.vision_ocr(served)
    if v is None:
        print("   D apple vision: pyobjc Vision framework not installed, skipped")
    else:
        for s, p in v:
            print(f"   D apple vision on served -> {s!r} (conf {p:.2f})")
            for r, c in cf.fuzzy_clue_match(s, clues):
                print(f"     clue match {r:.3f}: {str(c['clue'])[:60]!r}")

    print("\n== Exp 3: real captured chat line (truth: 'Cadava berries!') ==")
    native = a1.load_rgba(os.path.join(SCRATCH, "clue_native.png"))
    app = a1.app_view(native)
    region = app[1235:1275, 0:360]
    Image.fromarray(region[:, :, :3].astype(np.uint8)).save(os.path.join(SCRATCH, "exp3_chatline.png"))
    norm = cf.normalize_region(app, 0, 1235, 360, 40, SCALE)
    bg_est = tuple(np.median(norm[:, :, :3].reshape(-1, 3), axis=0))
    for fname in ["font_chat12", "font_chat14"]:
        font = a1.Font.load(os.path.join(cf.ASSETS, f"{fname}.fontmeta.json"))
        tmpl = cf.GlyphTemplates(font, (255, 255, 255), bg_est)
        best = None
        for y0 in range(font.basey + 2, norm.shape[0] - font.height + font.basey):
            ax, ay = anchor_search(norm, tmpl, 12, y0, rx=4, ry=0)
            res = cf.read_line_soft(norm, tmpl, ax, ay)
            if len(res["text"].strip()) >= 4 and (best is None or res["mean_zncc"] > best["mean_zncc"]):
                best = res
        show_read(f"soft OCR {fname} on normalized", best if best else "")

    v = cf.vision_ocr(region)
    if v is not None:
        for s, p in v:
            print(f"   apple vision on raw crop -> {s!r} (conf {p:.2f})")


if __name__ == "__main__":
    main()
