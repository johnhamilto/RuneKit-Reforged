"""Compass clue: read the needle bearing from the compass interface.

Ports the clue solver's needle estimation: centroid of dark needle pixels
inside the dial, refined by the mean angular offset over an annulus, then
interpolated through the calibration table mapping needle angles to world
directions. Full triangulation needs live position tracking; this reports
the direction to walk.
"""
import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from runekit import detection
from .assets import ClueAssets

logger = logging.getLogger(__name__)

REGION_W, REGION_H = 130, 170  # at needle position (north sprite - (53, -54))
CENTER = (65, 65)

# needle angle (raw) to world direction (dx east, dz north) calibration
SE_TABLE = [
    {"dx": -5, "dz": 0, "raw": 0.0454}, {"dx": -5, "dz": 1, "raw": 0.2037},
    {"dx": -5, "dz": 2, "raw": 0.359}, {"dx": -5, "dz": 3, "raw": 0.5237},
    {"dx": -5, "dz": 4, "raw": 0.6809}, {"dx": -5, "dz": 5, "raw": 0.8382},
    {"dx": -4, "dz": 5, "raw": 0.9919}, {"dx": -3, "dz": 5, "raw": 1.1473},
    {"dx": -2, "dz": 5, "raw": 1.3094}, {"dx": -1, "dz": 5, "raw": 1.4671},
    {"dx": 0, "dz": 5, "raw": 1.6218}, {"dx": 1, "dz": 5, "raw": 1.7809},
    {"dx": 2, "dz": 5, "raw": 1.9308}, {"dx": 3, "dz": 5, "raw": 2.0947},
    {"dx": 4, "dz": 5, "raw": 2.2519}, {"dx": 5, "dz": 5, "raw": 2.4098},
    {"dx": 5, "dz": 4, "raw": 2.5633}, {"dx": 5, "dz": 3, "raw": 2.7193},
    {"dx": 5, "dz": 2, "raw": 2.8785}, {"dx": 5, "dz": 1, "raw": -3.243},
    {"dx": 5, "dz": 0, "raw": -3.0895}, {"dx": 5, "dz": -1, "raw": -2.9294},
    {"dx": 5, "dz": -2, "raw": -2.7802}, {"dx": 5, "dz": -3, "raw": -2.6201},
    {"dx": 5, "dz": -4, "raw": -2.4627}, {"dx": 5, "dz": -5, "raw": -2.3011},
    {"dx": 4, "dz": -5, "raw": -2.147}, {"dx": 3, "dz": -5, "raw": -1.992},
    {"dx": 2, "dz": -6, "raw": -1.7791}, {"dx": 1, "dz": -5, "raw": -1.6773},
    {"dx": 0, "dz": -5, "raw": -1.5167}, {"dx": -1, "dz": -5, "raw": -1.3567},
    {"dx": -2, "dz": -5, "raw": -1.207}, {"dx": -3, "dz": -5, "raw": -1.0464},
    {"dx": -4, "dz": -5, "raw": -0.8892}, {"dx": -5, "dz": -5, "raw": -0.7311},
    {"dx": -5, "dz": -4, "raw": -0.577}, {"dx": -5, "dz": -3, "raw": -0.4213},
    {"dx": -5, "dz": -2, "raw": -0.2666}, {"dx": -5, "dz": -1, "raw": -0.1103},
]

WINDS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
         "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


@dataclass
class CompassRead:
    raw_angle: float  # screen-space needle angle
    direction: float  # world direction (atan2 of dz, dx), radians
    bearing_deg: float  # compass bearing, 0 = north, clockwise
    wind: str
    scale: float = 0.0


def _wrap(a: float, m: float) -> float:
    return (a % m + m) % m


def _angle_diff(a: float, b: float) -> float:
    return _wrap(b - a + math.pi, 2 * math.pi) - math.pi


def needle_angle(region: np.ndarray) -> Optional[float]:
    """Port of the bundle's two-pass dark-needle angle estimate. region must
    be the 1x compass area (REGION_W x REGION_H)."""
    rgb = region[:, :, :3].astype(np.int32)
    dark = (np.abs(rgb - np.array([19, 19, 18])).sum(axis=2) < 20) | (rgb < 5).all(axis=2)
    ys, xs = np.nonzero(dark)
    cx, cy = CENTER
    dx = xs - cx
    dy = ys - cy
    r2 = dx * dx + dy * dy
    inner = r2 < 60 * 60
    if not inner.any():
        return None
    d = float(dx[inner].mean())
    p = float(dy[inner].mean())
    m = math.atan2(p, d)

    rx = xs - cx - d
    ry = ys - cy - p
    rr = rx * rx + ry * ry
    ring = (rr > 20 * 20) & (rr < 50 * 50)
    if not ring.any():
        return m
    w = np.mod(np.arctan2(ry[ring], rx[ring]) - m, 2 * math.pi)
    w = np.where(w > 3 * math.pi / 2, w - 2 * math.pi, np.where(w > math.pi / 2, w - math.pi, w))
    return m + float(w.mean())


def table_direction(raw: float) -> float:
    lo, lo_d = None, -10.0
    hi, hi_d = None, 10.0
    for e in SE_TABLE:
        t = _angle_diff(raw, e["raw"])
        if t <= 0 and t > lo_d:
            lo, lo_d = e, t
        if t >= 0 and t < hi_d:
            hi, hi_d = e, t
    b = 1.0 if lo is hi else hi_d / (hi_d - lo_d)
    dz = b * lo["dz"] + (1 - b) * hi["dz"]
    dx = b * lo["dx"] + (1 - b) * hi["dx"]
    return math.atan2(-dz, -dx)


def read_compass(frame, assets: ClueAssets, hint: Optional[float] = None) -> Optional[CompassRead]:
    frame = detection.to_array(frame)
    north = detection.to_array(assets.needle("northimg"))
    m = detection.calibrate_scale(frame, north, hint=hint)
    if not m.ok:
        return None
    s = m.scale
    x0 = int(round(m.x - 53 * s))
    y0 = int(round(m.y + 54 * s))
    w = int(round(REGION_W * s))
    h = int(round(REGION_H * s))
    if x0 < 0 or y0 < 0 or x0 + w > frame.shape[1] or y0 + h > frame.shape[0]:
        return None
    region = detection.normalize_region(frame, x0, y0, w, h, s)
    if region.shape[0] != REGION_H or region.shape[1] != REGION_W:
        from PIL import Image

        im = Image.fromarray(region[:, :, :3].astype(np.uint8)).resize((REGION_W, REGION_H), Image.LANCZOS)
        region = detection.to_array(np.asarray(im))

    raw = needle_angle(region)
    if raw is None:
        logger.info("Compass found but needle not readable")
        return None
    direction = table_direction(raw)
    # world direction: +x east, +z north. Compass bearing clockwise from north.
    bearing = math.degrees(math.atan2(math.cos(direction), math.sin(direction)))
    bearing = _wrap(bearing, 360.0)
    wind = WINDS[int(round(bearing / 22.5)) % 16]
    logger.info("Compass needle raw %.3f -> direction %.3f, bearing %.0f (%s)", raw, direction, bearing, wind)
    return CompassRead(raw_angle=raw, direction=direction, bearing_deg=bearing, wind=wind, scale=s)
