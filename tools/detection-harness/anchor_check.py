"""Self-check for the clue interface anchors on synthetic frames: learning
picks only stable textured cells, matching tolerates one bad patch and
rejects shifts, the content key follows the body, the store round-trips.

    python anchor_check.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from runekit.cluehelper import anchors  # noqa: E402


def frame(rng, seed_strip, animate=True):
    """400x300 frame: textured static bar at y 40..64, animated world below."""
    img = np.zeros((300, 400, 3), dtype=np.uint8) + 30
    img[40:64, 60:340] = seed_strip
    img[100:300, :] = (
        rng.integers(0, 255, (200, 400, 3), dtype=np.uint8) if animate else 80
    )
    return img


def main() -> int:
    rng = np.random.default_rng(1)
    strip = rng.integers(0, 255, (24, 280, 3), dtype=np.uint8)
    frames = [frame(rng, strip) for _ in range(3)]
    region = (40, 30, 320, 260)  # covers the bar and part of the animated world
    anchor = anchors.learn("scroll", region, frames, 1.5, content=(40, 64, 320, 200))
    assert anchor is not None, "no anchor learned"
    assert len(anchor.patches) == anchors.MAX_PATCHES, len(anchor.patches)
    for p in anchor.patches:
        assert p.y < 64 and p.y + p.h > 40, f"patch misses the static bar: {p}"
        assert p.y + p.h <= 100, f"patch reaches into the animated area: {p}"
    xs = sorted(p.x for p in anchor.patches)
    assert all(b - a > anchors.CELL_W for a, b in zip(xs, xs[1:])), "patches touch"

    later = frame(rng, strip)
    assert anchor.present(later)
    shifted = np.roll(later, 3, axis=1)
    assert not anchor.present(shifted), "matched a 3 px shift"
    assert not anchor.present(
        frame(rng, rng.integers(0, 255, (24, 280, 3), dtype=np.uint8))
    ), "matched other content"
    damaged = later.copy()
    p = anchor.patches[0]
    damaged[p.y : p.y + p.h, p.x : p.x + p.w] = 255
    assert anchor.present(damaged), "one damaged patch should be tolerated"
    q = anchor.patches[1]
    damaged[q.y : q.y + q.h, q.x : q.x + q.w] = 255
    assert not anchor.present(damaged), "two damaged patches should fail"
    small = later[:50]
    assert not anchor.present(small), "frame too small must not match"

    body_a = frame(rng, strip, animate=False)
    body_b = body_a.copy()
    body_b[120:160, 100:300] = 200
    key_a, key_b = anchor.content_key(body_a), anchor.content_key(body_b)
    assert key_a.startswith("scroll:") and key_a != key_b, (key_a, key_b)
    assert anchor.content_key(body_a.copy()) == key_a
    static = anchors.learn("slide", region, frames, 1.0)
    assert static.content_key(body_a) == "slide"

    flat = [np.zeros((300, 400, 3), dtype=np.uint8) + 90 for _ in range(3)]
    assert anchors.learn("map", region, flat, 1.0) is None, "flat area must not anchor"
    animated = [rng.integers(0, 255, (300, 400, 3), dtype=np.uint8) for _ in range(3)]
    assert (
        anchors.learn("map", region, animated, 1.0) is None
    ), "animated area must not anchor"
    ring = anchors.learn("slide", region, frames, 1.0, exclude=(60, 40, 280, 24))
    assert ring is None, "excluding the bar leaves nothing stable"

    with tempfile.TemporaryDirectory() as tmp:
        store = anchors.AnchorStore(Path(tmp))
        assert store.all() == {} and all(v is None for v in store.summary().values())
        store.put(anchor)
        again = anchors.AnchorStore(Path(tmp))
        loaded = again.all()["scroll"]
        assert loaded.present(later) and loaded.content_key(body_a) == key_a
        assert (
            loaded.strip == anchor.strip
            and loaded.content == anchor.content
            and loaded.scale == 1.5
        )
        assert again.summary()["scroll"].startswith("Stored 1.50x")
        again.forget("scroll")
        assert again.all() == {} and anchors.AnchorStore(Path(tmp)).all() == {}
    print("anchor check ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
