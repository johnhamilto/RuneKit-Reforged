"""World map snapshots for solved clues.

Tiles come from the runeapps map service (the same one the clue solver's
web map uses) and are cached on disk. World coordinates map to pixels by
the map's CRS: px = (x + 0.5) * 2^zoom, py = (12799.5 - z) * 2^zoom,
512px tiles.
"""
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import requests
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

TILE_BASE = "https://runeapps-map.ams3.digitaloceanspaces.com/live"
TILE_SIZE = 512
MAX_ZOOM = 5
ORIGIN_Y = 12799.5


def _tile_path(cache_dir: Path, floor: int, zoom: int, tx: int, ty: int) -> Path:
    return cache_dir / "map_tiles" / f"{floor}" / f"{zoom}" / f"{tx}-{ty}.webp"


def _fetch_tile(cache_dir: Path, floor: int, zoom: int, tx: int, ty: int) -> Optional[Image.Image]:
    path = _tile_path(cache_dir, floor, zoom, tx, ty)
    if not path.exists():
        url = f"{TILE_BASE}/topdown-{floor}/{zoom}/{tx}-{ty}.webp"
        try:
            r = requests.get(url, timeout=10)
        except requests.RequestException:
            logger.warning("Map tile fetch failed: %s", url, exc_info=True)
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        # cache misses too (empty file) so oceans don't refetch every time
        path.write_bytes(r.content if r.status_code == 200 else b"")
    data = path.read_bytes()
    if not data:
        return None
    try:
        return Image.open(Path(path)).convert("RGB")
    except Exception:
        path.unlink(missing_ok=True)
        return None


def _to_px(x: float, z: float, zoom: int) -> Tuple[float, float]:
    scale = 2 ** zoom
    return (x + 0.5) * scale, (ORIGIN_Y - z) * scale


def _tile_shift(zoom: int) -> int:
    """The map's tile layer draws tiles offset 16 world tiles west and south
    (the source's half-variant-grid alignment); mirror it when compositing."""
    return 16 * 2 ** zoom


def _draw_marker(draw: ImageDraw.ImageDraw, cx: float, cy: float, main: bool = True):
    r = 9 if main else 6
    color = (230, 30, 30) if main else (230, 120, 20)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 255, 255), width=r // 2 + 3)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=r // 2 + 1)


def location_image(
    cache_dir: Path,
    spots: List[Tuple[int, int]],
    level: int = 0,
    size: Tuple[int, int] = (480, 330),
    primary: bool = True,
) -> Optional[np.ndarray]:
    """Map snapshot with the given world spots marked, or None if tiles are
    unavailable. With primary, the first spot is highlighted and centered."""
    if not spots:
        return None
    floor = level if 0 <= level <= 3 else 0
    w, h = size

    xs = [s[0] for s in spots]
    zs = [s[1] for s in spots]
    zoom = 3 if len(spots) == 1 else MAX_ZOOM
    while zoom > 0 and len(spots) > 1:
        span_x = (max(xs) - min(xs) + 24) * 2 ** zoom
        span_z = (max(zs) - min(zs) + 24) * 2 ** zoom
        if span_x <= w * 0.9 and span_z <= h * 0.9:
            break
        zoom -= 1

    if len(spots) == 1:
        cx, cy = _to_px(xs[0], zs[0], zoom)
    else:
        cx, cy = _to_px((max(xs) + min(xs)) / 2, (max(zs) + min(zs)) / 2, zoom)
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))

    shift = _tile_shift(zoom)
    tx_range = range((x0 + shift) // TILE_SIZE, (x0 + w + shift) // TILE_SIZE + 1)
    ty_range = range((y0 - shift) // TILE_SIZE, (y0 + h - shift) // TILE_SIZE + 1)

    def compose(layer_floor: int) -> int:
        count = 0
        for tx in tx_range:
            for ty in ty_range:
                tile = _fetch_tile(cache_dir, layer_floor, zoom, tx, ty)
                if tile is not None:
                    canvas.paste(tile, (tx * TILE_SIZE - shift - x0,
                                        ty * TILE_SIZE + shift - y0))
                    count += 1
        return count

    canvas = Image.new("RGB", (w, h), (24, 24, 24))
    found = compose(floor)
    if found == 0 and floor != 0:
        # upper floors are sparse; fall back to the ground layer for context
        found = compose(0)
    if found == 0:
        return None

    draw = ImageDraw.Draw(canvas)
    for i, (sx, sz) in enumerate(spots):
        px, py = _to_px(sx, sz, zoom)
        _draw_marker(draw, px - x0, py - y0, main=primary and i == 0)
    return np.asarray(canvas)
