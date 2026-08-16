"""Runtime download and cache of clue solver reference assets.

Needle sprites are extracted from the runeapps clue solver bundle and slide
puzzle tiles are fetched from runeapps directly. All of it is runeapps.org /
Jagex material: cached locally, never shipped with RuneKit.
"""
import base64
import io
import logging
import re
import time
from pathlib import Path
from typing import Dict

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)

BASE_URL = "https://runeapps.org"
BUNDLE_URL = BASE_URL + "/apps/clue/app.bundle.js"
CACHE_MAX_AGE = 30 * 24 * 3600

SLIDE_THEMES = {
    "b": "bandos", "t": "troll", "c": "castle", "r": "bridge", "o": "corp",
    "d": "dragon", "m": "maple", "s": "sliske", "gs": "guthixsword",
    "elf": "elf", "tusk": "tuska", "elderdrag": "elderdrag", "v": "v",
    "vyre": "vyre", "nomad": "nomad", "cit": "cit", "float": "float",
    "frost": "frost", "archer": "archer", "ara": "ara", "zam": "zam",
    "mage": "mage", "helw": "helwyr", "wolf": "wolf", "jas": "jas",
    "menn": "menn", "seal": "seal",
}


class ClueAssets:
    def __init__(self, cache_dir: Path):
        self.dir = cache_dir / "clue_assets"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _fetch(self, url: str, dest: Path) -> Path:
        fresh = dest.exists() and time.time() - dest.stat().st_mtime < CACHE_MAX_AGE
        if not fresh:
            try:
                req = requests.get(url, timeout=15)
                req.raise_for_status()
                dest.write_bytes(req.content)
                logger.info("Fetched %s", dest.name)
            except Exception:
                if not dest.exists():
                    raise
                logger.warning("Refresh of %s failed, using cache", url, exc_info=True)
        return dest

    def _extract_needles(self):
        marker = self.dir / ".needles_ok"
        if marker.exists():
            return
        src = self._fetch(BUNDLE_URL, self.dir / "app.bundle.js").read_text()

        mods = {}
        for m in re.finditer(
            r'(\d+)\(e,t,n\)\{\s*e\.exports=n\(\d+\)\.ImageDetect\.imageDataFromBase64\("([A-Za-z0-9+/=]+)"\)',
            src,
        ):
            mods[m.group(1)] = base64.b64decode(m.group(2))

        names: Dict[str, str] = {}
        for m in re.finditer(r"webpackImages\)\(\{([^}]*)\}\)", src, re.S):
            for name, mid in re.findall(r"(\w+):n\((\d+)\)", m.group(1)):
                names.setdefault(mid, name)

        count = 0
        for mid, raw in mods.items():
            name = names.get(mid)
            if not name:
                continue
            Image.open(io.BytesIO(raw)).save(self.dir / f"needle_{name}.png")
            count += 1
        if count < 10:
            raise RuntimeError(
                f"Only {count} needle sprites extracted; the runeapps bundle layout changed"
            )
        logger.info("Extracted %d needle sprites", count)
        marker.touch()

    def needle(self, name: str) -> np.ndarray:
        path = self.dir / f"needle_{name}.png"
        if not path.exists():
            self._extract_needles()
        return np.asarray(Image.open(path).convert("RGBA")).astype(np.int32)

    def _extract_fonts(self):
        """The bundle inlines its OCR fonts as JSON.parse('...') in a fixed
        order: smallcaps 9px, chatbox 12pt, chatbox 14pt."""
        import json

        src = self._fetch(BUNDLE_URL, self.dir / "app.bundle.js").read_text()
        names = ["font_allcaps9", "font_chat12", "font_chat14"]
        count = 0
        for name, m in zip(names, re.finditer(r"JSON\.parse\('(\{\"chars\".*?)'\)", src, re.S)):
            blob = m.group(1).encode().decode("unicode_escape")
            (self.dir / f"{name}.json").write_text(json.dumps(json.loads(blob)))
            count += 1
        if count != 3:
            raise RuntimeError(
                f"Expected 3 fonts in the runeapps bundle, found {count}; bundle layout changed"
            )

    def font(self, name: str) -> dict:
        import json

        path = self.dir / f"{name}.json"
        if not path.exists():
            self._extract_fonts()
        return json.loads(path.read_text())

    def slide_theme(self, letter: str) -> np.ndarray:
        filename = SLIDE_THEMES[letter]
        path = self._fetch(
            f"{BASE_URL}/apps/slidesolver/tileimgs/{filename}.png",
            self.dir / f"slide_{filename}.png",
        )
        return np.asarray(Image.open(path).convert("RGB")).astype(np.int32)
