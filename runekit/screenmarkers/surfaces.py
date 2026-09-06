"""Input surfaces for screen markers: a full-window canvas for drawing a new
marker and one small panel per marker for moving and resizing.

Both are frameless, translucent tool windows that never take keyboard focus,
so the game keeps receiving keys while the mouse works on a marker. On macOS
the window is also made a non-activating panel; otherwise the first click
would bring RuneKit to the front and the game would lose focus."""

import sys
from typing import Optional, Tuple

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QGuiApplication,
    QMouseEvent,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import QWidget

MARGIN = 8  # how far outside a marker's edge still grabs it, in points
HANDLE = 7
MIN_SIZE = 4

CURSORS = {
    (-1, -1): Qt.CursorShape.SizeFDiagCursor,
    (1, 1): Qt.CursorShape.SizeFDiagCursor,
    (1, -1): Qt.CursorShape.SizeBDiagCursor,
    (-1, 1): Qt.CursorShape.SizeBDiagCursor,
    (-1, 0): Qt.CursorShape.SizeHorCursor,
    (1, 0): Qt.CursorShape.SizeHorCursor,
    (0, -1): Qt.CursorShape.SizeVerCursor,
    (0, 1): Qt.CursorShape.SizeVerCursor,
    (0, 0): Qt.CursorShape.SizeAllCursor,
}


def label_font() -> QFont:
    return QFont("Verdana", 10, QFont.Weight.Bold)


def _keep_game_focused(widget: QWidget):
    if sys.platform != "darwin" or QGuiApplication.platformName() != "cocoa":
        return
    import AppKit
    import objc

    window = objc.objc_object(c_void_p=int(widget.winId())).window()
    window.setStyleMask_(
        int(window.styleMask()) | AppKit.NSWindowStyleMaskNonactivatingPanel
    )


class _Surface(QWidget):
    def __init__(self):
        super().__init__(
            None,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
            | Qt.WindowType.NoDropShadowWindowHint,
        )
        for attr in (
            Qt.WidgetAttribute.WA_TranslucentBackground,
            Qt.WidgetAttribute.WA_ShowWithoutActivating,
            Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow,
        ):
            self.setAttribute(attr, True)
        self.setMouseTracking(True)

    def showEvent(self, event):
        super().showEvent(event)
        _keep_game_focused(self)


class DrawSurface(_Surface):
    """Covers the game window while the user drags out a new marker."""

    drawn = Signal(QRect)
    cancelled = Signal()

    def __init__(self):
        super().__init__()
        self._pen = QPen()
        self._brush = QBrush()
        self._start: Optional[QPoint] = None
        self._rect = QRect()
        self.setCursor(Qt.CursorShape.CrossCursor)

    def begin(self, area: QRect, pen: QPen, brush: QBrush):
        self._pen = pen
        self._brush = brush
        self._start = None
        self._rect = QRect()
        self.setGeometry(area)
        self.show()
        self.raise_()

    def paintEvent(self, event):
        painter = QPainter(self)
        if self._rect.isValid():
            painter.setPen(self._pen)
            painter.setBrush(self._brush)
            painter.drawRect(self._rect)
        text = "Drag to draw a marker. Right-click to cancel."
        painter.setFont(QFont("Verdana", 11))
        box = painter.fontMetrics().boundingRect(text).adjusted(-12, -6, 12, 6)
        box.moveCenter(QPoint(self.width() // 2, 40))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 170))
        painter.drawRoundedRect(box, 6, 6)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._start = event.position().toPoint()
            self._rect = QRect(self._start, self._start)
            self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            self._start = None
            self.hide()
            self.cancelled.emit()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._start is not None:
            self._rect = QRect(self._start, event.position().toPoint()).normalized()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() != Qt.MouseButton.LeftButton or self._start is None:
            return
        rect = QRect(self._start, event.position().toPoint()).normalized()
        self._start = None
        self.hide()
        if rect.width() >= MIN_SIZE and rect.height() >= MIN_SIZE:
            self.drawn.emit(rect)
        else:
            self.cancelled.emit()


class MarkerPanel(_Surface):
    """One marker while editing: drag inside to move, drag an edge or corner
    to resize. The panel is the marker plus a grab margin, so clicks anywhere
    else still reach the game."""

    moved = Signal()  # the rect changed; emitted when the drag ends

    def __init__(self, marker, origin: QPoint):
        super().__init__()
        self.marker = marker
        self.origin = origin
        self.dragging = False
        self._mode: Tuple[int, int] = (0, 0)
        self._press = QPoint()
        self._start = QRect()
        self.sync()

    def sync(self):
        rect = self.marker.rect().translated(self.origin)
        self.setGeometry(rect.adjusted(-MARGIN, -MARGIN, MARGIN, MARGIN))
        self.update()

    def _inner(self) -> QRect:
        return QRect(MARGIN, MARGIN, self.marker.w, self.marker.h)

    def _hit(self, pos: QPoint) -> Tuple[int, int]:
        inner = self._inner()

        def side(value, low, high):
            near_low, near_high = abs(value - low), abs(value - high)
            if min(near_low, near_high) > MARGIN:
                return 0
            return -1 if near_low <= near_high else 1

        return (
            side(pos.x(), inner.left(), inner.right()),
            side(pos.y(), inner.top(), inner.bottom()),
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        inner = self._inner()
        painter.setPen(self.marker.pen())
        painter.setBrush(self.marker.brush())
        painter.drawRect(inner)
        if self.marker.label:
            color = QColor(self.marker.border)
            color.setAlpha(255)
            painter.setPen(color)
            painter.setFont(label_font())
            inset = self.marker.thickness // 2
            painter.drawText(
                inner.adjusted(inset + 3, inset + 2, 0, 0),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                self.marker.name,
            )
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.setBrush(QColor(255, 255, 255))
        xs = (inner.left(), inner.center().x(), inner.right())
        ys = (inner.top(), inner.center().y(), inner.bottom())
        half = HANDLE // 2
        for x in xs:
            for y in ys:
                if x == xs[1] and y == ys[1]:
                    continue
                painter.drawRect(x - half, y - half, HANDLE, HANDLE)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self.dragging = True
        self._mode = self._hit(event.position().toPoint())
        self._press = event.globalPosition().toPoint()
        self._start = self.marker.rect()
        self.setCursor(CURSORS[self._mode])

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self.dragging:
            self.setCursor(CURSORS[self._hit(event.position().toPoint())])
            return
        delta = event.globalPosition().toPoint() - self._press
        rect = QRect(self._start)
        ex, ey = self._mode
        if (ex, ey) == (0, 0):
            rect.translate(delta)
        else:
            if ex < 0:
                rect.setLeft(min(rect.left() + delta.x(), rect.right() - MIN_SIZE + 1))
            elif ex > 0:
                rect.setRight(max(rect.right() + delta.x(), rect.left() + MIN_SIZE - 1))
            if ey < 0:
                rect.setTop(min(rect.top() + delta.y(), rect.bottom() - MIN_SIZE + 1))
            elif ey > 0:
                rect.setBottom(
                    max(rect.bottom() + delta.y(), rect.top() + MIN_SIZE - 1)
                )
        self.marker.set_rect(rect)
        self.sync()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() != Qt.MouseButton.LeftButton or not self.dragging:
            return
        self.dragging = False
        self.moved.emit()
