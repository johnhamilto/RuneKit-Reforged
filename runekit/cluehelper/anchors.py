"""Learned pixel anchors for clue interfaces.

Each kind of interface (scroll, map, puzzle boxes) sits at a fixed place for
a given player. Once the user has solved one by hand we remember a few small
patches of its title bar or frame, taken from the live frame source, and
auto-detect only compares those patches on each frame. Nothing is learned
automatically: pressing Solve Clue on Screen with an interface open stores
or refreshes its kind, and Forget in the solver window drops it."""

import base64
import hashlib
import io
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

KINDS = ("scroll", "map", "compass", "slide", "lockbox", "towers", "knot")
LABELS = {
    "scroll": "Clue scroll",
    "map": "Treasure map",
    "compass": "Compass",
    "slide": "Slide puzzle",
    "lockbox": "Lockbox",
    "towers": "Towers",
    "knot": "Celtic knot",
}
# kinds whose content changes in place and should solve again when it does
HASH_CONTENT = {"scroll", "map", "knot"}

CELL_W, CELL_H = 20, 12
MAX_PATCHES = 6
MIN_PATCHES = 3
MIN_TEXTURE = 10.0  # grey std below which a cell is too flat to be distinctive
STABLE_TOL = 2  # per-channel change allowed across the learn frames
MATCH_MEAN = 6.0  # mean abs difference allowed for a patch to still match
MATCH_BIG = 40  # a pixel differing by more than this counts as changed
MATCH_FRAC = 0.05  # share of changed pixels allowed in a matching patch
MISSES_ALLOWED = 1  # one patch may be off (hover glow, tooltip)

Rect = Tuple[int, int, int, int]


def clip(rect: Rect, width: int, height: int) -> Rect:
    x, y, w, h = rect
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(width, int(x + w)), min(height, int(y + h))
    return (x0, y0, max(0, x1 - x0), max(0, y1 - y0))


def _intersects(a: Rect, b: Rect) -> bool:
    return (
        a[0] < b[0] + b[2]
        and b[0] < a[0] + a[2]
        and a[1] < b[1] + b[3]
        and b[1] < a[1] + a[3]
    )


@dataclass
class Patch:
    x: int
    y: int
    w: int
    h: int
    pixels: np.ndarray = field(repr=False)  # h x w x 3 uint8

    def matches(self, frame: np.ndarray) -> bool:
        if self.y + self.h > frame.shape[0] or self.x + self.w > frame.shape[1]:
            return False
        crop = frame[self.y : self.y + self.h, self.x : self.x + self.w, :3].astype(
            np.int16
        )
        diff = np.abs(crop - self.pixels.astype(np.int16)).max(axis=2)
        return (
            float(diff.mean()) <= MATCH_MEAN
            and float((diff > MATCH_BIG).mean()) <= MATCH_FRAC
        )


@dataclass
class Anchor:
    kind: str
    scale: float
    learned: str  # ISO timestamp
    strip: Rect  # where the patches were picked from
    content: Rect  # what content_key hashes
    patches: List[Patch]

    def present(self, frame: np.ndarray) -> bool:
        misses = 0
        for patch in self.patches:
            if not patch.matches(frame):
                misses += 1
                if misses > MISSES_ALLOWED:
                    return False
        return True

    def content_key(self, frame: np.ndarray) -> str:
        if self.kind not in HASH_CONTENT:
            return self.kind
        x, y, w, h = clip(self.content, frame.shape[1], frame.shape[0])
        if w < 8 or h < 8:
            return self.kind
        grey = cv2.cvtColor(
            np.ascontiguousarray(frame[y : y + h, x : x + w, :3]), cv2.COLOR_RGB2GRAY
        )
        small = cv2.resize(grey, (32, 24), interpolation=cv2.INTER_AREA) >> 5
        return (
            f"{self.kind}:{hashlib.blake2b(small.tobytes(), digest_size=6).hexdigest()}"
        )

    def to_json(self) -> dict:
        patches = []
        for p in self.patches:
            buf = io.BytesIO()
            Image.fromarray(p.pixels).save(buf, format="PNG")
            patches.append(
                {
                    "x": p.x,
                    "y": p.y,
                    "w": p.w,
                    "h": p.h,
                    "png": base64.b64encode(buf.getvalue()).decode(),
                }
            )
        return {
            "kind": self.kind,
            "scale": self.scale,
            "learned": self.learned,
            "strip": list(self.strip),
            "content": list(self.content),
            "patches": patches,
        }

    @classmethod
    def from_json(cls, data: dict) -> "Anchor":
        patches = []
        for p in data["patches"]:
            pixels = np.asarray(
                Image.open(io.BytesIO(base64.b64decode(p["png"]))).convert("RGB")
            )
            patches.append(
                Patch(int(p["x"]), int(p["y"]), int(p["w"]), int(p["h"]), pixels)
            )
        return cls(
            kind=str(data["kind"]),
            scale=float(data.get("scale", 0.0)),
            learned=str(data.get("learned", "")),
            strip=tuple(int(v) for v in data["strip"]),
            content=tuple(int(v) for v in data["content"]),
            patches=patches,
        )

    def summary(self) -> str:
        when = ""
        try:
            when = datetime.fromisoformat(self.learned).strftime("%-d %b %H:%M")
        except ValueError:
            pass
        scale = f"{self.scale:.2f}x, " if self.scale else ""
        return f"Stored {scale}{when}".rstrip(", ")


def learn(
    kind: str,
    strip: Rect,
    frames: Sequence[np.ndarray],
    scale: float,
    content: Optional[Rect] = None,
    exclude: Optional[Rect] = None,
) -> Optional[Anchor]:
    """Pick the most textured cells of strip that did not change across
    frames, spread out so no two touch. None when fewer than MIN_PATCHES
    qualify, which means the area is flat or animated."""
    if not frames or any(f.shape[:2] != frames[0].shape[:2] for f in frames):
        return None
    height, width = frames[0].shape[:2]
    x0, y0, w, h = clip(strip, width, height)
    candidates = []
    for cy in range(y0, y0 + h - CELL_H + 1, CELL_H):
        for cx in range(x0, x0 + w - CELL_W + 1, CELL_W):
            cell = (cx, cy, CELL_W, CELL_H)
            if exclude is not None and _intersects(cell, exclude):
                continue
            ref = frames[0][cy : cy + CELL_H, cx : cx + CELL_W, :3]
            stable = all(
                int(
                    np.abs(
                        f[cy : cy + CELL_H, cx : cx + CELL_W, :3].astype(np.int16)
                        - ref.astype(np.int16)
                    ).max()
                )
                <= STABLE_TOL
                for f in frames[1:]
            )
            if not stable:
                continue
            texture = float(ref.astype(np.float32).mean(axis=2).std())
            if texture < MIN_TEXTURE:
                continue
            candidates.append((texture, cx, cy))
    candidates.sort(reverse=True)
    chosen: List[Tuple[int, int]] = []
    for _, cx, cy in candidates:
        if any(abs(cx - ox) <= CELL_W and abs(cy - oy) <= CELL_H for ox, oy in chosen):
            continue
        chosen.append((cx, cy))
        if len(chosen) == MAX_PATCHES:
            break
    if len(chosen) < MIN_PATCHES:
        return None
    patches = [
        Patch(
            cx,
            cy,
            CELL_W,
            CELL_H,
            np.ascontiguousarray(
                frames[0][cy : cy + CELL_H, cx : cx + CELL_W, :3]
            ).copy(),
        )
        for cx, cy in chosen
    ]
    return Anchor(
        kind=kind,
        scale=float(scale or 0.0),
        learned=datetime.now().isoformat(timespec="seconds"),
        strip=(x0, y0, w, h),
        content=tuple(int(v) for v in (content or (x0, y0, w, h))),
        patches=patches,
    )


class AnchorStore:
    """The anchors on disk, one per interface kind, safe to read from the
    screening thread while the solve thread stores a new one."""

    def __init__(self, cache_dir: Path):
        self.path = Path(cache_dir) / "clue_anchors.json"
        self._lock = threading.Lock()
        self._anchors: Dict[str, Anchor] = {}
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            self._anchors = {
                k: Anchor.from_json(v) for k, v in data.items() if k in KINDS
            }
        except (ValueError, KeyError, TypeError, OSError):
            logger.warning(
                "Ignoring unreadable clue anchors at %s", self.path, exc_info=True
            )

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({k: a.to_json() for k, a in self._anchors.items()})
        )

    def all(self) -> Dict[str, Anchor]:
        with self._lock:
            return dict(self._anchors)

    def put(self, anchor: Anchor):
        with self._lock:
            self._anchors[anchor.kind] = anchor
            self._save()

    def forget(self, kind: str):
        with self._lock:
            if kind in self._anchors:
                del self._anchors[kind]
                self._save()

    def summary(self) -> Dict[str, Optional[str]]:
        with self._lock:
            return {
                kind: (self._anchors[kind].summary() if kind in self._anchors else None)
                for kind in KINDS
            }
