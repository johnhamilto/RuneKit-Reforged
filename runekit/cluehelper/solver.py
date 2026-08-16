"""Clue recognition: OCR the game frame, find the clue interface by its title,
and solve what the text allows: database text clues, coordinate clues, and
scan area identification.

Databases are fetched at runtime and cached locally; they are runeapps.org
data and are never shipped with RuneKit.
"""
import difflib
import json
import logging
import re
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
COORDS_URL = "https://runeapps.org/apps/clue/coords.json"
CACHE_MAX_AGE = 7 * 24 * 3600

INTERFACE_TITLES = ("mysterious clue scroll", "treasure map")
# clue types whose on-screen text can be matched against the database
TEXT_TYPES = ("simple", "cryptic", "anagram", "emote", "action")

SOLVED_RATIO = 0.75
LOW_RATIO = 0.55

# The in-game coordinate system: 00 degrees 00 minutes is world tile
# (2440, 3161) and one minute of arc is 1/1.875 tiles.
COORD_ORIGIN_X = 2440
COORD_ORIGIN_Z = 3161
COORD_TILES_PER_MINUTE = 1 / 1.875
COORD_RE = re.compile(
    r"(\d{1,2})\s*degrees?[,.\s]*(\d{1,2})\s*minutes?[,.\s]*(north|south)"
    r"[,.\s]*(\d{1,2})\s*degrees?[,.\s]*(\d{1,2})\s*minutes?[,.\s]*(east|west)",
    re.IGNORECASE,
)


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


def _fetch_json(url: str, cache_path: Path):
    fresh = cache_path.exists() and time.time() - cache_path.stat().st_mtime < CACHE_MAX_AGE
    if not fresh:
        try:
            req = requests.get(url, timeout=10)
            req.raise_for_status()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(req.content)
            logger.info("Updated %s", cache_path.name)
        except Exception:
            if not cache_path.exists():
                raise
            logger.warning("Refresh of %s failed, using cache", url, exc_info=True)

    return json.loads(cache_path.read_text())


def load_databases(cache_dir: Path) -> dict:
    return {
        "clues": _fetch_json(CLUES_URL, cache_dir / "clue_db.json"),
        "coords": _fetch_json(COORDS_URL, cache_dir / "coord_db.json"),
    }


def clue_text(entry: dict) -> str:
    clue = entry.get("scantext") if entry.get("type") == "scan" else entry.get("clue")
    if isinstance(clue, list):
        clue = " ".join(str(s) for s in clue)
    return clue or ""


def describe(entry: dict) -> str:
    kind = entry.get("type", "unknown")
    parts = [f"Type: {kind}"]
    if kind == "coordinate":
        where = f"Dig at: x={entry['x']}, z={entry['z']}"
        if entry.get("level"):
            where += f", level {entry['level']}"
        parts.append(where)
        if not entry.get("known_spot"):
            parts.append("Note: no known dig spot at this exact tile, computed from the scroll text")
        return "\n".join(parts)

    if kind == "scan":
        parts.append(f"Scan area: {entry.get('scan', 'unknown')}")
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


def parse_coordinates(text: str) -> Optional[dict]:
    # OCR sometimes reads 0 as O inside the number tokens
    cleaned = re.sub(r"\b([0-9O]{1,2})\b", lambda m: m.group(1).replace("O", "0"), text)
    m = COORD_RE.search(cleaned)
    if not m:
        return None
    latdeg, latmin, ns, longdeg, longmin, ew = m.groups()
    ns_sign = 1 if ns.lower() == "north" else -1
    ew_sign = 1 if ew.lower() == "east" else -1
    x = COORD_ORIGIN_X + round(ew_sign * (60 * int(longdeg) + int(longmin)) * COORD_TILES_PER_MINUTE)
    z = COORD_ORIGIN_Z + round(ns_sign * (60 * int(latdeg) + int(latmin)) * COORD_TILES_PER_MINUTE)
    return {"x": x, "z": z, "text": m.group(0)}


def snap_to_dig_spot(x: int, z: int, coords: List[dict]) -> Tuple[Optional[dict], int]:
    best, best_d = None, 10 ** 9
    for spot in coords:
        d = max(abs(spot["x"] - x), abs(spot["z"] - z))
        if d < best_d:
            best, best_d = spot, d
    return best, best_d


def solve_frame(frame, dbs: dict, ocr=vision.ocr_lines) -> SolveResult:
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

    full_text = " ".join(l["text"] for l in body).strip()

    coord = parse_coordinates(full_text)
    if coord:
        entry = {"type": "coordinate", "x": coord["x"], "z": coord["z"]}
        spot, dist = snap_to_dig_spot(coord["x"], coord["z"], dbs["coords"])
        if spot is not None and dist <= 3:
            entry.update(x=spot["x"], z=spot["z"], level=spot.get("level"), known_spot=True)
        result.status = "solved"
        result.read_text = coord["text"]
        result.matches = [(1.0 if entry.get("known_spot") else 0.9, entry)]
        return result

    # The clue text is the top run of lines; anything below it (tooltips, game
    # UI) hurts the match. Score every prefix and keep the best one.
    candidates = [
        e for e in dbs["clues"]
        if (e.get("type") in TEXT_TYPES or e.get("type") == "scan") and clue_text(e)
    ]
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
