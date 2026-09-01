"""Clue data from the RuneScape Wiki.

The wiki stores every clue as structured rows in its public Bucket API:
canonical clue text, solution, travel routes, requirements, and world
coordinates. Rows are synced into a local cache and used as the primary
match/guidance source; the data is fetched at runtime and never shipped.
"""
import html
import json
import logging
import re
import time
from pathlib import Path
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

API = "https://runescape.wiki/api.php"
USER_AGENT = "RuneKit-Reforged clue helper (https://github.com/johnhamilto/RuneKit-Reforged)"
CACHE_MAX_AGE = 7 * 24 * 3600
PAGE_SIZE = 500
FIELDS = (
    "page_name", "clue_type", "difficulty", "clue", "title", "solution",
    "requirements", "location", "travel", "x_coordinate", "y_coordinate",
    "plane", "type_data",
)
TYPES = ("simple", "cryptic", "anagram", "emote", "coordinate", "scan", "map", "compass")


def plain(s: Optional[str]) -> str:
    """Wikitext/HTML to readable text."""
    if not s:
        return ""
    # skill requirement gadgets carry the skill name only in attributes
    s = re.sub(
        r'<span[^>]*class="skillreq"[^>]*data-skill="([^"]+)"[^>]*data-level="(\d+)"[^>]*>.*?</span>',
        r"\2 \1", s, flags=re.S)
    s = re.sub(r"<sup[^>]*ordinal-suffix[^>]*>(.*?)</sup>", r"\1", s, flags=re.S)
    s = re.sub(r"<sup[^>]*>.*?</sup>", "", s, flags=re.S)
    # the floor-convention gadget duplicates text for the US variant
    prev = None
    while prev != s:
        prev = s
        s = re.sub(
            r'<span[^>]*class="[^"]*(?:noexcerpt|floornumber-us)[^"]*"[^>]*>[^<>]*</span>',
            "", s)
    s = re.sub(r"<[^>]+>", "", s)
    prev = None
    while prev != s:  # innermost templates first
        prev = s
        s = re.sub(r"\{\{[^{}]*\}\}", "", s)
    prev = None
    while prev != s:  # innermost links first; resolves nested file links too
        prev = s
        s = re.sub(r"\[\[[^\[\]|]*\|([^\[\]]*)\]\]", r"\1", s)
        s = re.sub(r"\[\[([^\[\]]*)\]\]", r"\1", s)
    s = s.replace("'''", "").replace("''", "")
    s = html.unescape(s)
    s = re.sub(r"\bfloor floor\b", "floor", s)  # floor-convention gadget residue
    lines = []
    for line in s.split("\n"):
        line = re.sub(r"^[*#:]+\s*", "• ", line.strip())
        line = re.sub(r"\s+", " ", line)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _bucket_query(lua: str) -> List[dict]:
    r = requests.get(
        API,
        params={"action": "bucket", "format": "json", "formatversion": "2", "query": lua},
        headers={"User-Agent": USER_AGENT},
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()
    rows = data.get("bucket")
    if not isinstance(rows, list):
        raise RuntimeError(f"Bucket query failed: {json.dumps(data)[:300]}")
    return rows


def _fetch_rows() -> List[dict]:
    sel = ",".join(f"'{f}'" for f in FIELDS)
    rows = []
    for clue_type in TYPES:
        offset = 0
        while True:
            page = _bucket_query(
                f"bucket('clue').select({sel})"
                f".where({{'clue_type','{clue_type}'}})"
                f".limit({PAGE_SIZE}).offset({offset}).run()"
            )
            rows.extend(page)
            if len(page) < PAGE_SIZE:
                break
            offset += PAGE_SIZE
    return rows


def _normalize(row: dict) -> dict:
    xs = row.get("x_coordinate") or []
    zs = row.get("y_coordinate") or []
    type_data = {}
    if row.get("type_data"):
        try:
            type_data = json.loads(row["type_data"])
        except ValueError:
            pass
    entry = {
        "source": "wiki",
        "type": row.get("clue_type"),
        "difficulty": row.get("difficulty"),
        "clue": plain(row.get("clue")),
        "title": plain(row.get("title")).split("\n")[0],
        "solution": plain(row.get("solution")),
        "requirements": plain(row.get("requirements")),
        "location": plain(row.get("location")),
        "travel": plain(row.get("travel")),
        "fight": type_data.get("fight") or "",
        "group": type_data.get("group") or "",
        "page_name": row.get("page_name") or "",
        "level": row.get("plane") or 0,
        "items": [i["item"] for i in type_data.get("items") or [] if i.get("item")],
        "emotes": type_data.get("emotes") or [],
    }
    hole = type_data.get("hideyhole")
    if isinstance(hole, dict):
        entry["hideyhole"] = {
            "x": hole.get("x"),
            "z": hole.get("y"),
            "text": plain(hole.get("description") or ""),
        }
    if xs and zs:
        entry["x"], entry["z"] = xs[0], zs[0]
        if len(xs) > 1:
            entry["scan_spots"] = [
                {"x": x, "z": z} for x, z in zip(xs[1:], zs[1:])
            ]
    return entry


CACHE_FORMAT = 2


def load(cache_dir: Path) -> List[dict]:
    """Synced wiki clue entries; empty list when offline with no cache."""
    cache_path = cache_dir / "wiki_clues.json"
    data = None
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text())
        except ValueError:
            cache_path.unlink(missing_ok=True)
    fresh = (
        data is not None
        and isinstance(data, dict)
        and data.get("v") == CACHE_FORMAT
        and time.time() - cache_path.stat().st_mtime < CACHE_MAX_AGE
    )
    if not fresh:
        try:
            entries = [_normalize(r) for r in _fetch_rows()]
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({"v": CACHE_FORMAT, "entries": entries}))
            logger.info("Synced %d clue entries from the wiki", len(entries))
            return entries
        except Exception:
            logger.warning("Wiki clue sync failed", exc_info=True)
    if isinstance(data, dict):
        return data.get("entries") or []
    if isinstance(data, list):  # stale pre-versioned cache; usable offline
        return data
    return []


def nearest(entries: List[dict], x: int, z: int, types=(), max_dist: int = 15) -> Optional[dict]:
    """Wiki entry whose target is closest to (x, z) within max_dist tiles."""
    best, best_d = None, max_dist + 1
    for e in entries:
        if types and e.get("type") not in types:
            continue
        if e.get("x") is None:
            continue
        d = max(abs(e["x"] - x), abs(e["z"] - z))
        if d < best_d:
            best, best_d = e, d
    return best
