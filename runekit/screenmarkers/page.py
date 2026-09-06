"""Settings tab listing the screen markers."""

import sys
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QGuiApplication, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import EDIT_KEY, Marker, ScreenMarkers

if TYPE_CHECKING:
    from runekit.host import Host

COLUMNS = ("Show", "Name", "Border", "Fill", "Width", "Label")
if sys.platform == "darwin":
    HINT = f"Hold {EDIT_KEY} over the game to move a marker or drag its edges and corners to resize it."
else:
    HINT = "Moving and resizing markers with the mouse is not available on this platform yet."


class MarkersPage(QWidget):
    def __init__(self, host: "Host", **kwargs):
        super().__init__(**kwargs)
        self.markers: ScreenMarkers = host.screen_markers
        self._building = False
        self._layout()
        self.markers.changed.connect(self.refresh)
        self.refresh()

    def _layout(self):
        layout = QHBoxLayout(self)
        self.setLayout(layout)

        left = QVBoxLayout()
        hint = QLabel(HINT, self)
        hint.setWordWrap(True)
        left.addWidget(hint)

        self.table = QTableWidget(0, len(COLUMNS), self)
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.itemChanged.connect(self.on_item_changed)
        left.addWidget(self.table, 1)
        layout.addLayout(left, 1)

        buttons = QVBoxLayout()
        buttons.setAlignment(Qt.AlignmentFlag.AlignTop)
        for text, slot in (
            ("Add", self.markers.add_marker),
            ("Remove", self.on_remove),
            ("Import", self.on_import),
            ("Export", self.on_export),
        ):
            button = QPushButton(text, self)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        layout.addLayout(buttons)

    @Slot()
    def refresh(self):
        self._building = True
        self.table.setRowCount(0)
        self.table.setRowCount(len(self.markers.markers))
        for row, marker in enumerate(self.markers.markers):
            self.table.setCellWidget(row, 0, self._check(marker, "visible"))
            item = QTableWidgetItem(marker.name)
            item.setData(Qt.ItemDataRole.UserRole, marker.id)
            self.table.setItem(row, 1, item)
            self.table.setCellWidget(row, 2, self._color_button(marker, "border"))
            self.table.setCellWidget(row, 3, self._color_button(marker, "fill"))
            width = QSpinBox(self.table)
            width.setRange(1, 10)
            width.setValue(marker.thickness)
            width.valueChanged.connect(
                lambda value, m=marker: self._set(m, "thickness", value)
            )
            self.table.setCellWidget(row, 4, width)
            self.table.setCellWidget(row, 5, self._check(marker, "label"))
        self._building = False

    def _set(self, marker: Marker, attr: str, value):
        setattr(marker, attr, value)
        self.markers.commit(notify=False)

    def _check(self, marker: Marker, attr: str) -> QWidget:
        holder = QWidget(self.table)
        layout = QHBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        box = QCheckBox(holder)
        box.setChecked(getattr(marker, attr))
        box.toggled.connect(lambda on, m=marker: self._set(m, attr, on))
        layout.addWidget(box)
        return holder

    def _color_button(self, marker: Marker, attr: str) -> QPushButton:
        button = QPushButton(self.table)
        self._paint_button(button, QColor(getattr(marker, attr)))

        def pick():
            color = QColorDialog.getColor(
                QColor(getattr(marker, attr)),
                self,
                f"{attr.title()} color",
                QColorDialog.ColorDialogOption.ShowAlphaChannel,
            )
            if color.isValid():
                self._set(marker, attr, color.name(QColor.NameFormat.HexArgb))
                self._paint_button(button, color)

        button.clicked.connect(pick)
        return button

    @staticmethod
    def _paint_button(button: QPushButton, color: QColor):
        pixmap = QPixmap(18, 18)
        pixmap.fill(color)
        button.setIcon(QIcon(pixmap))
        button.setToolTip(color.name(QColor.NameFormat.HexArgb))

    def _selected(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        item = self.table.item(rows[0].row(), 1)
        return self._by_id(item.data(Qt.ItemDataRole.UserRole))

    def _by_id(self, marker_id: str):
        for marker in self.markers.markers:
            if marker.id == marker_id:
                return marker
        return None

    @Slot(QTableWidgetItem)
    def on_item_changed(self, item: QTableWidgetItem):
        if self._building or item.column() != 1:
            return
        marker = self._by_id(item.data(Qt.ItemDataRole.UserRole))
        if marker is None:
            return
        name = item.text().strip()
        if not name:
            self._building = True
            item.setText(marker.name)
            self._building = False
            return
        self._set(marker, "name", name)

    @Slot()
    def on_remove(self):
        marker = self._selected()
        if marker is None:
            return
        confirm = QMessageBox.question(self, "Remove marker", f"Remove {marker.name}?")
        if confirm == QMessageBox.StandardButton.Yes:
            self.markers.remove(marker)

    @Slot()
    def on_export(self):
        QGuiApplication.clipboard().setText(self.markers.export_json())
        QMessageBox.information(
            self,
            "Export markers",
            f"Copied {len(self.markers.markers)} markers to the clipboard",
        )

    @Slot()
    def on_import(self):
        try:
            count = self.markers.import_json(QGuiApplication.clipboard().text())
        except (ValueError, KeyError, TypeError) as e:
            QMessageBox.warning(
                self, "Import markers", f"The clipboard does not hold markers: {e}"
            )
            return
        QMessageBox.information(self, "Import markers", f"Added {count} markers")
