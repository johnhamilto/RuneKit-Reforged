"""Clue solver window: solve button, live status, rendered results."""
import logging
from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QLabel,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

BOARD_RENDER = 280


def np_to_pixmap(arr: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(arr[:, :, :3].astype(np.uint8))
    img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img.copy())


def _draw_number(painter: QPainter, x: float, y: float, text: str,
                 color=QColor(255, 220, 80), size=13):
    font = QFont("Verdana", size, QFont.Weight.Bold)
    painter.setFont(font)
    painter.setPen(QPen(QColor(0, 0, 0)))
    for dx, dy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        painter.drawText(int(x + dx), int(y + dy), text)
    painter.setPen(QPen(color))
    painter.drawText(int(x), int(y), text)


def render_slide(board: List[int], moves: List[int], theme_img: np.ndarray,
                 count: int) -> QPixmap:
    """Compose the read board from the reference theme and number the moves."""
    ref = 49
    tiles = theme_img.shape[1] // ref
    canvas = np.zeros((5 * ref, 5 * ref, 3), dtype=np.uint8)
    for cell in range(25):
        r, c = divmod(cell, 5)
        part = board[cell]
        if part == 24:
            canvas[r * ref:(r + 1) * ref, c * ref:(c + 1) * ref] = (16, 14, 18)
        else:
            pr, pc = divmod(part, tiles)
            canvas[r * ref:(r + 1) * ref, c * ref:(c + 1) * ref] = \
                theme_img[pr * ref:(pr + 1) * ref, pc * ref:(pc + 1) * ref, :3]

    pix = np_to_pixmap(canvas).scaled(
        BOARD_RENDER, BOARD_RENDER,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    tile = BOARD_RENDER / 5
    painter = QPainter(pix)
    visits = {}
    for i, cell in enumerate(moves[:count]):
        r, c = divmod(cell, 5)
        n = visits.get(cell, 0)
        visits[cell] = n + 1
        cx = (c + 0.5) * tile + (n % 3 - 1) * 15 - 6
        cy = (r + 0.5) * tile + (n // 3 % 3 - 1) * 15 + 5
        _draw_number(painter, cx, cy, str(i + 1))
    painter.end()
    return pix


def render_lockbox(grid: List[List[int]], presses: List[List[int]],
                   needles: List[np.ndarray]) -> QPixmap:
    tile = 38
    canvas = np.zeros((5 * tile, 5 * tile, 3), dtype=np.uint8)
    canvas[:, :] = (44, 38, 32)
    for r in range(5):
        for c in range(5):
            n = needles[grid[r][c]]
            a = n[:, :, 3:4] / 255.0
            y0, x0 = r * tile, c * tile
            canvas[y0:y0 + tile, x0:x0 + tile] = (
                n[:, :, :3] * a + canvas[y0:y0 + tile, x0:x0 + tile] * (1 - a)
            ).astype(np.uint8)

    pix = np_to_pixmap(canvas).scaled(
        BOARD_RENDER, BOARD_RENDER,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    cell = BOARD_RENDER / 5
    painter = QPainter(pix)
    for r in range(5):
        for c in range(5):
            if presses[r][c]:
                _draw_number(painter, (c + 0.5) * cell - 5, (r + 0.5) * cell + 6,
                             str(presses[r][c]), size=15)
    painter.end()
    return pix


def render_towers(grid: List[List[int]], filled: List[List[int]],
                  top: List[int], bot: List[int],
                  left: List[int], right: List[int]) -> QPixmap:
    cell = 40
    size = cell * 7
    pix = QPixmap(size, size)
    pix.fill(QColor(30, 26, 22))
    painter = QPainter(pix)
    painter.setPen(QPen(QColor(90, 80, 66)))
    for i in range(6):
        painter.drawLine(cell, cell + i * cell, cell * 6, cell + i * cell)
        painter.drawLine(cell + i * cell, cell, cell + i * cell, cell * 6)
    for i in range(5):
        x = cell * (i + 1.5) - 5
        _draw_number(painter, x, cell * 0.75, str(top[i]), QColor(255, 205, 10))
        _draw_number(painter, x, cell * 6.6, str(bot[i]), QColor(255, 205, 10))
        y = cell * (i + 1.5) + 6
        _draw_number(painter, cell * 0.4, y, str(left[i]), QColor(255, 205, 10))
        _draw_number(painter, cell * 6.3, y, str(right[i]), QColor(255, 205, 10))
    for r in range(5):
        for c in range(5):
            color = QColor(255, 255, 255) if filled[r][c] else QColor(120, 235, 120)
            _draw_number(painter, cell * (c + 1.5) - 5, cell * (r + 1.5) + 6,
                         str(grid[r][c]), color, size=15)
    painter.end()
    return pix.scaled(BOARD_RENDER, BOARD_RENDER,
                      Qt.AspectRatioMode.KeepAspectRatio,
                      Qt.TransformationMode.SmoothTransformation)


KNOT_RING_COLORS = {
    0: QColor(90, 120, 235),
    1: QColor(235, 180, 40),
    2: QColor(90, 90, 100),
    3: QColor(200, 200, 205),
}


def render_knot(paths, intersections, offsets=None) -> QPixmap:
    """Diagram of the read knot: one diamond per slot, colored per ring,
    rune ids as numbers, crossings split-colored."""
    from PySide6.QtGui import QBrush, QPolygonF
    from PySide6.QtCore import QPointF

    half = 11
    pts = [(s.x, s.y) for p in paths for s in p]
    if not pts:
        return QPixmap(1, 1)
    xs = [half * (t - n) for t, n in pts]
    ys = [-half * (t + n) for t, n in pts]
    x0, y0 = min(xs) - 2 * half, min(ys) - 2 * half
    w = int(max(xs) - x0 + 2 * half)
    h = int(max(ys) - y0 + 2 * half)
    pix = QPixmap(w, h)
    pix.fill(QColor(30, 26, 22))
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    crossing_at = {(i["x"], i["y"]): i for i in intersections}

    def center(t, n):
        return (half * (t - n) - x0, -half * (t + n) - y0)

    for ring, path in enumerate(paths):
        for i, slot in enumerate(path):
            cx, cy = center(slot.x, slot.y)
            inter = crossing_at.get((slot.x, slot.y))
            is_under = inter is not None and inter.get("col2") == ring and inter.get("col1") != ring
            if is_under:
                continue  # drawn by the ring on top
            diamond = QPolygonF([
                QPointF(cx, cy - half), QPointF(cx + half, cy),
                QPointF(cx, cy + half), QPointF(cx - half, cy),
            ])
            painter.setPen(QPen(QColor(15, 13, 11), 1))
            painter.setBrush(QBrush(KNOT_RING_COLORS.get(ring, QColor(150, 150, 150))))
            painter.drawPolygon(diamond)
            if inter is not None and inter.get("col2") is not None:
                other = inter["col2"] if inter.get("col1") == ring else inter.get("col1")
                painter.setBrush(QBrush(KNOT_RING_COLORS.get(other, QColor(150, 150, 150))))
                painter.drawPolygon(QPolygonF([
                    QPointF(cx, cy + half), QPointF(cx - half, cy), QPointF(cx, cy - half),
                ]))
            rune = slot.rune
            if rune >= 0:
                _draw_number(painter, cx - 4, cy + 4, str(rune), QColor(255, 255, 255), size=8)
            elif rune != -10000:
                _draw_number(painter, cx - 4, cy + 4, "?", QColor(255, 160, 160), size=8)
    painter.end()
    return pix


def render_compass(bearing_deg: float) -> QPixmap:
    import math

    size = 160
    pix = QPixmap(size, size)
    pix.fill(QColor(30, 26, 22))
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    cx = cy = size / 2
    painter.setPen(QPen(QColor(90, 80, 66), 2))
    painter.drawEllipse(int(cx - 60), int(cy - 60), 120, 120)
    _draw_number(painter, cx - 5, cy - 62, "N", QColor(200, 200, 210), size=11)
    rad = math.radians(bearing_deg)
    tip = (cx + 52 * math.sin(rad), cy - 52 * math.cos(rad))
    back = (cx - 18 * math.sin(rad), cy + 18 * math.cos(rad))
    painter.setPen(QPen(QColor(255, 90, 60), 4))
    painter.drawLine(int(back[0]), int(back[1]), int(tip[0]), int(tip[1]))
    left = (tip[0] - 12 * math.sin(rad + 0.5), tip[1] + 12 * math.cos(rad + 0.5))
    right = (tip[0] - 12 * math.sin(rad - 0.5), tip[1] + 12 * math.cos(rad - 0.5))
    painter.drawLine(int(tip[0]), int(tip[1]), int(left[0]), int(left[1]))
    painter.drawLine(int(tip[0]), int(tip[1]), int(right[0]), int(right[1]))
    painter.end()
    return pix


class ClueSolverWindow(QWidget):
    solve_requested = Signal()
    auto_toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Clue Solver")
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setMinimumWidth(320)

        layout = QVBoxLayout(self)
        self.solve_button = QPushButton("Solve Clue on Screen")
        self.solve_button.setMinimumHeight(40)
        font = self.solve_button.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 2)
        self.solve_button.setFont(font)
        self.solve_button.clicked.connect(self.solve_requested)
        layout.addWidget(self.solve_button)

        self.auto_check = QCheckBox("Detect clues automatically")
        self.auto_check.toggled.connect(self.auto_toggled)
        layout.addWidget(self.auto_check)

        self.status = QLabel("Open a clue or puzzle in game, then solve.")
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.image = QLabel()
        self.image.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.image.hide()
        layout.addWidget(self.image)

        self.body = QTextBrowser()
        self.body.setOpenExternalLinks(False)
        self.body.setMinimumHeight(120)
        self.body.hide()
        layout.addWidget(self.body)
        layout.addStretch(1)

    def set_busy(self, busy: bool, text: Optional[str] = None):
        self.solve_button.setEnabled(not busy)
        if text is not None:
            self.status.setText(text)

    def show_message(self, text: str, details: str = ""):
        self.set_busy(False)
        self.status.setText(text)
        self.image.hide()
        if details:
            self.body.setHtml(f"<small style='color: gray'>{details}</small>")
            self.body.show()
        else:
            self.body.hide()

    def show_result(self, status: str, body_html: str,
                    pixmap: Optional[QPixmap] = None, details: str = ""):
        self.set_busy(False)
        self.status.setText(status)
        if pixmap is not None:
            self.image.setPixmap(pixmap)
            self.image.show()
        else:
            self.image.hide()
        html = body_html
        if details:
            html += f"<hr><small style='color: gray'>{details}</small>"
        self.body.setHtml(html)
        self.body.show()

    def open(self):
        self.show()
        self.raise_()
        self.activateWindow()
