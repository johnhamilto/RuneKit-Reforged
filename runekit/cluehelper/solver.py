"""Clue recognition: OCR the game frame, find the clue interface by its title,
and match the clue text against the runeapps clue database.

The database is fetched at runtime and cached locally; it is runeapps.org
data and is never shipped with RuneKit.
"""
import difflib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import requests
from PIL import Image

from . import vision

logger = logging.getLogger(__name__)

CLUES_URL = "https://runeapps.org/apps/clue/clues.json"
CACHE_MAX_AGE = 7 * 24 * 3600

INTERFACE_TITLES = ("mysterious clue scroll", "treasure map")
# clue types whose on-screen text can be matched against the database
TEXT_TYPES = ("simple", "cryptic", "anagram", "emote", "action")

SOLVED_RATIO = 0.75
LOW_RATIO = 0.55


@dataclass
class SolveResult:
    status: str  # solved | low_confidence | unsupported | no_clue
    title: str = ""
    read_text: str = ""
    matches: List[Tuple[float, dict]] = field(default_factory=list)
    lines: List[vision.OcrLine] = field(default_factory=list)

    @property
    def best(self) -> Optional[Tuple[float, dict]]:
        return self.matches[0] if self.matches else None


def load_clue_db(cache_path: Path) -> List[dict]:
    fresh = cache_path.exists() and time.time() - cache_path.stat().st_mtime < CACHE_MAX_AGE
    if not fresh:
        try:
            req = requests.get(CLUES_URL, timeout=10)
            req.raise_for_status()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(req.content)
            logger.info("Clue database updated from %s", CLUES_URL)
        except Exception:
            if not cache_path.exists():
                raise
            logger.warning("Clue database refresh failed, using cache", exc_info=True)

    return json.loads(cache_path.read_text())


def clue_text(entry: dict) -> str:
    clue = entry.get("clue") or ""
    if isinstance(clue, list):
        clue = " ".join(str(s) for s in clue)
    return clue


def describe(entry: dict) -> str:
    parts = [f"Type: {entry.get('type', 'unknown')}"]
    answer = entry.get("answer")
    if answer:
        parts.append(f"Answer: {answer}")
    if entry.get("x") is not None:
        parts.append(f"Location: x={entry['x']}, z={entry['z']}")
    comment = entry.get("//comment")
    if comment:
        parts.append(f"Note: {comment}")
    return "\n".join(parts)


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _to_image(frame) -> Image.Image:
    if isinstance(frame, np.ndarray):
        return Image.fromarray(frame[:, :, :3].astype(np.uint8))
    return frame


def solve_frame(frame, db: List[dict], ocr=vision.ocr_lines) -> SolveResult:
    lines = ocr(_to_image(frame))

    title = None
    for line in lines:
        if max(_ratio(line["text"], t) for t in INTERFACE_TITLES) >= 0.7:
            title = line
            break

    if title is None:
        return SolveResult(status="no_clue", lines=lines)

    # clue body: lines below the title, roughly centered on it. Vision boxes
    # are normalized with origin at the bottom left.
    tx, ty, tw, th = title["box"]
    tcx = tx + tw / 2
    body = []
    for line in lines:
        if line is title:
            continue
        x, y, w, h = line["box"]
        below = y + h <= ty + th * 0.5
        near = abs((x + w / 2) - tcx) < 0.3 and y > ty - 0.35
        if below and near:
            body.append(line)

    body.sort(key=lambda l: (-l["box"][1], l["box"][0]))

    result = SolveResult(status="unsupported", title=title["text"], lines=lines)
    if not body:
        return result

    # The clue text is the top run of lines; anything below it (tooltips, game
    # UI) hurts the match. Score every prefix and keep the best one.
    candidates = [e for e in db if e.get("type") in TEXT_TYPES and clue_text(e)]
    for k in range(1, len(body) + 1):
        read = " ".join(l["text"] for l in body[:k]).strip()
        scored = sorted(((_ratio(read, clue_text(e)), e) for e in candidates), key=lambda t: -t[0])
        if not result.matches or scored[0][0] > result.matches[0][0]:
            result.matches = scored[:3]
            result.read_text = read

    best = result.matches[0][0] if result.matches else 0.0
    if best >= SOLVED_RATIO:
        result.status = "solved"
    elif best >= LOW_RATIO:
        result.status = "low_confidence"
    return result
