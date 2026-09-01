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
from . import wiki

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
    map_image: Optional[np.ndarray] = None

    @property
    def best(self) -> Optional[Tuple[float, dict]]:
        return self.matches[0] if self.matches else None


def entry_spots(entry: dict) -> Tuple[List[Tuple[int, int]], int, bool]:
    """World spots to mark for a matched entry: (spots, level, primary).
    primary means the first spot is the single target."""
    spots = []
    primary = False
    if entry.get("x") is not None and entry.get("z") is not None:
        spots.append((entry["x"], entry["z"]))
        primary = True
    for s in entry.get("scan_spots") or []:
        spots.append((s["x"], s["z"]))
    hole = entry.get("hideyhole") or {}
    if hole.get("x") is not None and hole.get("z") is not None:
        spots.append((hole["x"], hole["z"]))
    return spots, int(entry.get("level") or 0), primary


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
        # a cache written by a different app version must degrade, not crash
        "wiki": [e for e in wiki.load(cache_dir) if isinstance(e, dict)],
    }


def clue_text(entry: dict) -> str:
    if entry.get("type") in ("img", "emptyimg"):
        return ""  # the clue field holds an image signature, not text
    if entry.get("source") == "wiki" and entry.get("type") != "scan":
        return entry.get("clue") or ""
    clue = entry.get("scantext") if entry.get("type") == "scan" else entry.get("clue")
    if isinstance(clue, list):
        clue = " ".join(str(s) for s in clue)
    return clue or ""


def describe(entry: dict) -> str:
    kind = entry.get("type", "unknown")
    if entry.get("source") == "wiki":
        head = f"Type: {kind}"
        if entry.get("difficulty"):
            head += f" ({entry['difficulty']})"
        parts = [head]
        if entry.get("emotes"):
            parts.append(f"Emote: {' then '.join(entry['emotes'])}")
        if entry.get("items"):
            parts.append(f"Items: {', '.join(entry['items'])}")
        if entry.get("hideyhole", {}).get("text"):
            parts.append(entry["hideyhole"]["text"])
        if entry.get("solution"):
            parts.append(f"Solution: {entry['solution']}")
        if entry.get("location"):
            parts.append(f"Location: {entry['location']}")
        if entry.get("travel"):
            travel = entry["travel"]
            parts.append(f"Travel: {travel}" if "\n" not in travel else f"Travel:\n{travel}")
        req = entry.get("requirements") or ""
        if req and req.lower() != "none":
            parts.append(f"Requires: {req}")
        if entry.get("fight"):
            parts.append(f"Ambush: {entry['fight']}")
        if entry.get("x") is not None:
            where = f"World spot: x={entry['x']}, z={entry['z']}"
            if entry.get("level"):
                where += f", level {entry['level']}"
            parts.append(where)
        return "\n".join(parts)
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
        spots = entry.get("scan_spots") or []
        if spots:
            listed = ", ".join(f"({s['x']},{s['z']})" for s in spots[:8])
            more = f" and {len(spots) - 8} more" if len(spots) > 8 else ""
            parts.append(f"Dig spots ({len(spots)}): {listed}{more}")
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


def _matchkey(s: str) -> str:
    """Case- and punctuation-insensitive comparison key. OCR drops or invents
    punctuation freely, and the databases disagree on it for the same clue."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", "", s.lower())).strip()


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

    # clue body: lines below the title. Scroll text is centered in the modal,
    # so real body lines sit on the title's center; the gate scales with the
    # title width so it tracks the modal size at any window size. Vision boxes
    # are normalized with origin at the bottom left.
    tx, ty, tw, th = title["box"]
    tcx = tx + tw / 2
    body = []
    for line in lines:
        if line is title:
            continue
        x, y, w, h = line["box"]
        below = y + h <= ty + th * 0.5
        near = abs((x + w / 2) - tcx) < max(tw, 0.05) and y > ty - 0.35
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
        wiki_entry = wiki.nearest(
            dbs.get("wiki") or [], coord["x"], coord["z"], types=("coordinate",), max_dist=3
        )
        if wiki_entry is not None:
            entry = dict(wiki_entry, known_spot=True)
        else:
            spot, dist = snap_to_dig_spot(coord["x"], coord["z"], dbs["coords"])
            if spot is not None and dist <= 3:
                entry.update(x=spot["x"], z=spot["z"], level=spot.get("level"), known_spot=True)
        result.status = "solved"
        result.read_text = coord["text"]
        result.matches = [(1.0 if entry.get("known_spot") else 0.9, entry)]
        return result

    # The clue is one contiguous run of lines; neighbouring UI text (task
    # lists, tooltips) that slips into the body hurts the match. Score every
    # span of up to six lines and keep the best one. Wiki entries carry the
    # richer answer, so they win ties against the runeapps copy of a clue.
    candidates = [
        e for e in dbs["clues"]
        if (e.get("type") in TEXT_TYPES or e.get("type") == "scan") and clue_text(e)
    ] + [
        e for e in dbs.get("wiki") or []
        if e.get("type") in TEXT_TYPES and e.get("clue")
    ]
    matchers = [
        (difflib.SequenceMatcher(None, "", _matchkey(clue_text(e))), e)
        for e in candidates
    ]
    for i in range(len(body)):
        for j in range(i + 1, min(i + 6, len(body)) + 1):
            read = " ".join(l["text"] for l in body[i:j]).strip()
            key = _matchkey(read)
            scored = []
            for m, e in matchers:
                m.set_seq1(key)
                scored.append((m.ratio(), e))
            scored.sort(key=lambda t: (-round(t[0], 3), 0 if t[1].get("source") == "wiki" else 1))
            if not result.matches or scored[0][0] > result.matches[0][0]:
                result.matches = scored[:3]
                result.read_text = read

    best = result.matches[0][0] if result.matches else 0.0
    if best >= SOLVED_RATIO:
        result.status = "solved"
    elif best >= LOW_RATIO:
        result.status = "low_confidence"

    if result.matches and result.matches[0][1].get("type") == "scan":
        entry = dict(result.matches[0][1])
        entry["scan_spots"] = [
            c for c in dbs["coords"] if c.get("clueid") == entry.get("scan")
        ]
        result.matches[0] = (result.matches[0][0], entry)
    return result
