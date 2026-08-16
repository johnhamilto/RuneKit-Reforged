"""Download the runeapps clue solver bundle and extract its detection assets
(needle sprites, OCR font definitions, clue database) into ./assets.

These assets are runeapps.org/Jagex property; they are fetched locally for
debugging and must not be committed or redistributed.
"""
import base64
import io
import json
import os
import re

import requests
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("CLUE_ASSETS") or os.path.join(HERE, "assets")
BASE = "https://runeapps.org/apps/clue/"

os.makedirs(OUT, exist_ok=True)

for name in ["clues.json", "coords.json"]:
    open(os.path.join(OUT, name), "wb").write(requests.get(BASE + name).content)
    print("downloaded", name)

src = requests.get(BASE + "app.bundle.js").text
print(f"downloaded app.bundle.js ({len(src)} bytes)")

# needle sprites: webpack modules exporting ImageDetect.imageDataFromBase64(...)
mods = {}
for m in re.finditer(r'(\d+)\(e,t,n\)\{\s*e\.exports=n\(\d+\)\.ImageDetect\.imageDataFromBase64\("([A-Za-z0-9+/=]+)"\)', src):
    mods[m.group(1)] = Image.open(io.BytesIO(base64.b64decode(m.group(2))))

names = {}
for m in re.finditer(r'webpackImages\)\(\{([^}]*)\}\)', src, re.S):
    for name, mid in re.findall(r'(\w+):n\((\d+)\)', m.group(1)):
        names.setdefault(mid, name)

for mid, im in sorted(mods.items(), key=lambda kv: int(kv[0])):
    name = names.get(mid, f"mod{mid}")
    im.save(os.path.join(OUT, f"needle_{name}.png"))
print(f"extracted {len(mods)} needles")

# fontmeta JSONs are inlined as JSON.parse('...') with JS string escaping.
# Order in the bundle: aa_9px_mono_allcaps, chatbox 12pt, chatbox 14pt.
fontnames = ["font_allcaps9", "font_chat12", "font_chat14"]
count = 0
for name, m in zip(fontnames, re.finditer(r"JSON\.parse\('(\{\"chars\".*?)'\)", src, re.S)):
    blob = m.group(1).encode().decode("unicode_escape")
    data = json.loads(blob)
    json.dump(data, open(os.path.join(OUT, f"{name}.fontmeta.json"), "w"))
    count += 1
    print(f"extracted {name}: {len(data['chars'])} glyphs")
if count != 3:
    raise SystemExit(f"expected 3 fonts, found {count}: bundle layout changed, update the regexes")
