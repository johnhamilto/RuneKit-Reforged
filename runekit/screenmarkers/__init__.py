"""Persistent rectangles over the game window, in the spirit of RuneLite's
screen markers. Markers live in game window coordinates, so they follow the
window around; they show only while the game or RuneKit is in front."""

import json
import logging
import sys
import uuid
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional, TYPE_CHECKING

from PySide6.QtCore import QObject, QRect, QRectF, QSettings, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QGuiApplication, QPen
from PySide6.QtWidgets import (
    QGraphicsItemGroup,
    QGraphicsRectItem,
    QGraphicsSimpleTextItem,
    QMessageBox,
)

from .surfaces import DrawSurface, MarkerPanel, label_font

if TYPE_CHECKING:
    from runekit.game import GameInstance

logger = logging.getLogger(__name__)

MARKERS_KEY = "screenmarkers/markers"
SHOWN_KEY = "screenmarkers/shown"
DEFAULT_BORDER = "#ffffff00"  # #AARRGGBB, opaque yellow
DEFAULT_FILL = "#00ffff00"  # fully transparent
DEFAULT_THICKNESS = 3
POLL_MS = 3000
EDIT_KEY = "Option" if sys.platform == "darwin" else "Alt"


@dataclass
class Marker:
    name: str
    x: int
    y: int
    w: int
    h: int
    border: str = DEFAULT_BORDER
    fill: str = DEFAULT_FILL
    thickness: int = DEFAULT_THICKNESS
    visible: bool = True
    label: bool = False
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def rect(self) -> QRect:
        return QRect(self.x, self.y, self.w, self.h)

    def set_rect(self, rect: QRect):
        self.x, self.y, self.w, self.h = rect.x(), rect.y(), rect.width(), rect.height()

    def pen(self) -> QPen:
        pen = QPen(QColor(self.border))
        pen.setWidth(self.thickness)
        pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        return pen

    def brush(self) -> QBrush:
        return QBrush(QColor(self.fill))

    @classmethod
    def from_dict(cls, data: dict) -> "Marker":
        marker = cls(
            name=str(data["name"]),
            x=int(data["x"]),
            y=int(data["y"]),
            w=int(data["w"]),
            h=int(data["h"]),
            border=str(data.get("border", DEFAULT_BORDER)),
            fill=str(data.get("fill", DEFAULT_FILL)),
            thickness=int(data.get("thickness", DEFAULT_THICKNESS)),
            visible=bool(data.get("visible", True)),
            label=bool(data.get("label", False)),
            id=str(data.get("id") or uuid.uuid4().hex),
        )
        if not QColor(marker.border).isValid() or not QColor(marker.fill).isValid():
            raise ValueError(f"bad colour on marker {marker.name!r}")
        return marker


class ScreenMarkers(QObject):
    changed = Signal()  # markers were added, removed, or edited

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.instance_provider: Optional[Callable[[], Optional["GameInstance"]]] = None
        self.markers: List[Marker] = []
        self.shown = QSettings().value(SHOWN_KEY, True, bool)
        self.pinned = False  # edit mode held on from the tray menu
        self._alt = False  # edit mode while Alt/Option is held
        self._instance = None
        self._group = None
        self._panels: List[MarkerPanel] = []
        self._draw: Optional[DrawSurface] = None
        self._load()
        QGuiApplication.instance().applicationStateChanged.connect(
            self._update_visibility
        )
        self._poll = QTimer(self)
        self._poll.setInterval(POLL_MS)
        self._poll.timeout.connect(self._tick)
        self._poll.start()

    # -------------------------------------------------------------- storage

    def _load(self):
        raw = QSettings().value(MARKERS_KEY, "", str)
        if not raw:
            return
        try:
            self.markers = [Marker.from_dict(item) for item in json.loads(raw)]
        except (ValueError, KeyError, TypeError):
            logger.warning("Ignoring unreadable screen markers: %.200r", raw)

    def _save(self):
        QSettings().setValue(MARKERS_KEY, json.dumps([asdict(m) for m in self.markers]))

    def commit(self, notify: bool = True):
        """Persist and redraw after markers were edited in place."""
        self._save()
        self._acquire()
        self._render()
        self._apply_edit()
        if notify:
            self.changed.emit()

    def remove(self, marker: Marker):
        self.markers = [m for m in self.markers if m is not marker]
        self.commit()

    def export_json(self) -> str:
        return json.dumps([asdict(m) for m in self.markers], indent=2)

    def import_json(self, text: str) -> int:
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("expected a list of markers")
        new = [Marker.from_dict(item) for item in data]
        for marker in new:
            marker.id = uuid.uuid4().hex
        self.markers.extend(new)
        self.commit()
        return len(new)

    @Slot(bool)
    def set_shown(self, shown: bool):
        self.shown = shown
        QSettings().setValue(SHOWN_KEY, shown)
        self._update_visibility()
        self._apply_edit()

    # -------------------------------------------------------------- drawing

    @Slot()
    def add_marker(self):
        instance = self._acquire()
        if instance is None:
            QMessageBox.critical(
                None,
                "No game instances found",
                "Cannot find open RuneScape window. Launch the game before adding a marker",
            )
            return
        if self._draw is None:
            self._draw = DrawSurface()
            self._draw.drawn.connect(self._on_drawn)
            self._draw.cancelled.connect(self._apply_edit)
        self._close_panels()
        style = Marker("", 0, 0, 0, 0)
        self._draw.begin(instance.get_position(), style.pen(), style.brush())

    @Slot(QRect)
    def _on_drawn(self, rect: QRect):
        name = f"Marker {len(self.markers) + 1}"
        self.markers.append(
            Marker(name, rect.x(), rect.y(), rect.width(), rect.height())
        )
        self.commit()

    # -------------------------------------------------------------- editing

    @Slot(bool)
    def on_alt(self, down: bool):
        self._alt = down
        self._apply_edit()

    @Slot(bool)
    def pin_editing(self, on: bool):
        self.pinned = on
        self._apply_edit()

    def _apply_edit(self, *_):
        drawing = self._draw is not None and self._draw.isVisible()
        want = (self._alt or self.pinned) and self.shown and not drawing
        if not want:
            if self._panels and not any(p.dragging for p in self._panels):
                self._close_panels()
            return
        wanted = [m for m in self.markers if m.visible]
        if [p.marker for p in self._panels] == wanted:
            return
        if any(p.dragging for p in self._panels):
            return
        self._close_panels()
        instance = self._acquire()
        if instance is None or not wanted:
            return
        origin = instance.get_position().topLeft()
        for marker in wanted:
            panel = MarkerPanel(marker, origin)
            panel.moved.connect(self._on_panel_moved)
            panel.show()
            self._panels.append(panel)
        self._update_visibility()

    def _close_panels(self):
        for panel in self._panels:
            panel.hide()
            panel.deleteLater()
        self._panels = []
        self._update_visibility()

    @Slot()
    def _on_panel_moved(self):
        self.commit()

    # ------------------------------------------------------------- overlay

    def _find(self) -> Optional["GameInstance"]:
        instance = self.instance_provider() if self.instance_provider else None
        if instance is not None:
            try:
                instance.get_position()
            except Exception:
                return None  # the window is gone even if the manager still lists it
        return instance

    def _acquire(self) -> Optional["GameInstance"]:
        instance = self._find()
        if instance is not self._instance:
            self._attach(instance)
        return self._instance

    def _attach(self, instance):
        if self._instance is not None:
            for signal, slot in (
                (self._instance.focusChanged, self._update_visibility),
                (self._instance.positionChanged, self._on_moved),
            ):
                try:
                    signal.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass
        self._instance = instance
        if instance is not None:
            instance.focusChanged.connect(self._update_visibility)
            instance.positionChanged.connect(self._on_moved)
        self._render()

    @Slot()
    def _tick(self):
        self._acquire()
        if self._instance is not None and self._group is None and self.markers:
            self._render()

    @Slot(QRect)
    def _on_moved(self, rect: QRect):
        for panel in self._panels:
            panel.origin = rect.topLeft()
            panel.sync()

    def _render(self):
        if self._group is not None:
            try:
                scene = self._group.scene()
                if scene is not None:
                    scene.removeItem(self._group)
            except RuntimeError:
                pass  # the overlay already dropped it with the game window
            self._group = None
        if self._instance is None or not self.markers:
            return
        try:
            area = self._instance.get_overlay_area()
        except NotImplementedError:
            return
        group = QGraphicsItemGroup(area)
        for marker in self.markers:
            if not marker.visible:
                continue
            rect = QGraphicsRectItem(QRectF(marker.rect()), group)
            rect.setPen(marker.pen())
            rect.setBrush(marker.brush())
            if marker.label:
                color = QColor(marker.border)
                color.setAlpha(255)
                text = QGraphicsSimpleTextItem(marker.name, group)
                text.setFont(label_font())
                text.setBrush(QBrush(color))
                text.setPen(QPen(QColor(0, 0, 0), 0.5))
                inset = marker.thickness // 2
                text.setPos(marker.x + inset + 3, marker.y + inset + 2)
        self._group = group
        self._update_visibility()

    def _update_visibility(self, *_):
        if self._group is None:
            return
        in_front = (self._instance is not None and self._instance.is_focused()) or (
            QGuiApplication.applicationState() == Qt.ApplicationState.ApplicationActive
        )
        self._group.setVisible(self.shown and not self._panels and in_front)
