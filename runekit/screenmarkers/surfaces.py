"""The canvas for drawing a new marker: a frameless, translucent tool window
covering the game while the user drags out the rectangle. Clicking it brings
RuneKit to the front, which is fine for this one explicit action. Editing
existing markers goes through the platform input hook instead, so no RuneKit
window is ever under the mouse and the game keeps focus."""

from typing import Optional

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .geometry import MIN_SIZE


def label_font() -> QFont:
    return QFont("Verdana", 10, QFont.Weight.Bold)


class DrawSurface(QWidget):
    drawn = Signal(QRect)
    cancelled = Signal()

    def __init__(self):
        super().__init__(
            None,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.NoDropShadowWindowHint,
        )
        for attr in (
            Qt.WidgetAttribute.WA_TranslucentBackground,
            Qt.WidgetAttribute.WA_ShowWithoutActivating,
            Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow,
        ):
            self.setAttribute(attr, True)
        self.setMouseTracking(True)
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
