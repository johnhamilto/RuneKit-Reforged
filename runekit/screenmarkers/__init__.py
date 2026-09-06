"""Persistent rectangles over the game window, in the spirit of RuneLite's
screen markers. Markers live in game window coordinates, so they follow the
window around; they show only while the game or RuneKit is in front.

Editing never puts a RuneKit window under the mouse: macOS treats an
Option-click on another app's window as an app switch and hides the game.
Instead the platform manager intercepts mouse events while editing is on and
offers them to on_mouse, which drags or removes the marker and redraws the
click-through overlay."""

import json
import logging
import sys
import uuid
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

from PySide6.QtCore import (
    QObject,
    QPoint,
    QRect,
    QRectF,
    QSettings,
    QTimer,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtGui import QBrush, QColor, QGuiApplication, QPen
from PySide6.QtWidgets import (
    QGraphicsItemGroup,
    QGraphicsRectItem,
    QGraphicsSimpleTextItem,
    QMessageBox,
)

from .geometry import HANDLE, dragged_rect, hit_side
from .surfaces import DrawSurface, label_font

if TYPE_CHECKING:
    from runekit.game import GameInstance, GameManager

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

    def __init__(self, manager: "GameManager", parent=None):
        super().__init__(parent=parent)
        self.instance_provider: Optional[Callable[[], Optional["GameInstance"]]] = None
        self.markers: List[Marker] = []
        self.shown = QSettings().value(SHOWN_KEY, True, bool)
        self.pinned = False  # edit mode held on from the tray menu
        self._manager = manager
        self._alt = False  # edit mode while Alt/Option is held
        self._editing = False
        self._drag = None  # (marker, mode, start point, start rect) during a drag
        self._origin: Optional[QPoint] = None  # game window top-left on screen
        self._instance = None
        self._group = None
        self._items: Dict[str, dict] = {}
        self._draw: Optional[DrawSurface] = None
        self._load()
        manager.mouse_hook = self.on_mouse
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
        style = Marker("", 0, 0, 0, 0)
        self._draw.begin(instance.get_position(), style.pen(), style.brush())
        self._apply_edit()

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
        if want == self._editing:
            return
        if not want and self._drag is not None:
            return  # finish the drag first; on_mouse comes back here on release
        if want:
            instance = self._acquire()
            if instance is None:
                return
            self._origin = instance.get_position().topLeft()
        self._editing = want
        logger.debug("Marker editing %s", "on" if want else "off")
        self._manager.set_mouse_capture(want)
        self._render()

    def on_mouse(self, kind: str, x: float, y: float) -> bool:
        """Mouse event from the platform while capture is on; True swallows it.
        kind is "down", "drag" or "up" for the left button, "right" for a right press.
        """
        if kind in ("down", "right"):
            if (
                not self._editing
                or self._origin is None
                or self._group is None
                or not self._group.isVisible()
                or (kind == "right" and self._drag is not None)
            ):
                return False
            local = self._local(x, y)
            for marker in reversed(self.markers):
                if not marker.visible:
                    continue
                mode = hit_side(local, marker.rect())
                if mode is None:
                    continue
                if kind == "right":
                    logger.info("Removed marker %r with a right-click", marker.name)
                    self.remove(marker)
                else:
                    self._drag = (marker, mode, local, marker.rect())
                return True
            return False
        if self._drag is None:
            return False
        marker, mode, start, start_rect = self._drag
        marker.set_rect(dragged_rect(start_rect, mode, self._local(x, y) - start))
        self._place(marker)
        if kind == "up":
            self._drag = None
            self.commit()
        return True

    def _local(self, x: float, y: float) -> QPoint:
        return QPoint(int(x) - self._origin.x(), int(y) - self._origin.y())

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
        self._apply_edit()

    @Slot(QRect)
    def _on_moved(self, rect: QRect):
        self._origin = rect.topLeft()

    def _render(self):
        if self._group is not None:
            try:
                scene = self._group.scene()
                if scene is not None:
                    scene.removeItem(self._group)
            except RuntimeError:
                pass  # the overlay already dropped it with the game window
            self._group = None
        self._items = {}
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
            rect = QGraphicsRectItem(group)
            rect.setPen(marker.pen())
            rect.setBrush(marker.brush())
            entry = {"rect": rect, "label": None, "handles": []}
            if marker.label:
                color = QColor(marker.border)
                color.setAlpha(255)
                text = QGraphicsSimpleTextItem(marker.name, group)
                text.setFont(label_font())
                text.setBrush(QBrush(color))
                text.setPen(QPen(QColor(0, 0, 0), 0.5))
                entry["label"] = text
            if self._editing:
                for _ in range(8):
                    handle = QGraphicsRectItem(group)
                    handle.setPen(QPen(QColor(0, 0, 0), 1))
                    handle.setBrush(QBrush(QColor(255, 255, 255)))
                    entry["handles"].append(handle)
            self._items[marker.id] = entry
            self._place(marker)
        self._group = group
        self._update_visibility()

    def _place(self, marker: Marker):
        entry = self._items.get(marker.id)
        if entry is None:
            return
        rect = marker.rect()
        entry["rect"].setRect(QRectF(rect))
        if entry["label"] is not None:
            inset = marker.thickness // 2
            entry["label"].setPos(rect.x() + inset + 3, rect.y() + inset + 2)
        if entry["handles"]:
            xs = (rect.left(), rect.center().x(), rect.right())
            ys = (rect.top(), rect.center().y(), rect.bottom())
            spots = [(x, y) for x in xs for y in ys if (x, y) != (xs[1], ys[1])]
            half = HANDLE / 2
            for handle, (x, y) in zip(entry["handles"], spots):
                handle.setRect(x - half, y - half, HANDLE, HANDLE)

    def _update_visibility(self, *_):
        if self._group is None:
            return
        in_front = (self._instance is not None and self._instance.is_focused()) or (
            QGuiApplication.applicationState() == Qt.ApplicationState.ApplicationActive
        )
        self._group.setVisible(self.shown and in_front)
