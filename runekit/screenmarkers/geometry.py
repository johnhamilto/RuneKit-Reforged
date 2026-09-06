"""Hit testing and drag arithmetic for marker editing, in game window points."""

from typing import Optional, Tuple

from PySide6.QtCore import QPoint, QRect

MARGIN = 8  # how far outside a marker's edge still grabs it
HANDLE = 7
MIN_SIZE = 4

# (-1 left/top, 0 inside, 1 right/bottom) per axis; (0, 0) moves the marker
Mode = Tuple[int, int]


def hit_side(pos: QPoint, rect: QRect) -> Optional[Mode]:
    if not rect.adjusted(-MARGIN, -MARGIN, MARGIN, MARGIN).contains(pos):
        return None

    def side(value, low, high):
        near_low, near_high = abs(value - low), abs(value - high)
        if min(near_low, near_high) > MARGIN:
            return 0
        return -1 if near_low <= near_high else 1

    return side(pos.x(), rect.left(), rect.right()), side(
        pos.y(), rect.top(), rect.bottom()
    )


def dragged_rect(start: QRect, mode: Mode, delta: QPoint) -> QRect:
    rect = QRect(start)
    ex, ey = mode
    if (ex, ey) == (0, 0):
        rect.translate(delta)
        return rect
    if ex < 0:
        rect.setLeft(min(rect.left() + delta.x(), rect.right() - MIN_SIZE + 1))
    elif ex > 0:
        rect.setRight(max(rect.right() + delta.x(), rect.left() + MIN_SIZE - 1))
    if ey < 0:
        rect.setTop(min(rect.top() + delta.y(), rect.bottom() - MIN_SIZE + 1))
    elif ey > 0:
        rect.setBottom(max(rect.bottom() + delta.y(), rect.top() + MIN_SIZE - 1))
    return rect
