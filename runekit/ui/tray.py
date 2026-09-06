from typing import TYPE_CHECKING

from PySide6.QtCore import Slot, Signal, QCoreApplication
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QSystemTrayIcon, QMenu

if TYPE_CHECKING:
    from runekit.host import Host


class TrayIcon(QSystemTrayIcon):
    on_settings = Signal()
    on_solve_clue = Signal()
    on_add_marker = Signal()
    on_show_markers = Signal(bool)
    on_edit_markers = Signal(bool)

    host: "Host"

    def __init__(self, host: "Host"):
        global _tray_icon
        super().__init__(QIcon(":/runekit/ui/trayicon.png"))

        _tray_icon = self

        self.host = host
        self._setup_menu()
        self.setContextMenu(self.menu)

    def _setup_menu(self):
        if not hasattr(self, "menu"):
            self.menu = QMenu("RuneKit")

        self.menu.clear()

        self._setup_app_menu("", self.menu)

        self.menu.addSeparator()
        self.menu_solve_clue = self.menu.addAction("Solve Clue on Screen")
        self.menu_solve_clue.triggered.connect(self.on_solve_clue)
        self.menu.addSeparator()
        self.menu_add_marker = self.menu.addAction("Add Screen Marker")
        self.menu_add_marker.triggered.connect(self.on_add_marker)
        self.menu_show_markers = self.menu.addAction("Show Screen Markers")
        self.menu_show_markers.setCheckable(True)
        self.menu_show_markers.setChecked(self.host.screen_markers.shown)
        self.menu_show_markers.toggled.connect(self.on_show_markers)
        self.menu_edit_markers = self.menu.addAction("Edit Screen Markers")
        self.menu_edit_markers.setCheckable(True)
        self.menu_edit_markers.setChecked(self.host.screen_markers.pinned)
        self.menu_edit_markers.toggled.connect(self.on_edit_markers)
        self.menu.addSeparator()
        self.menu_settings = self.menu.addAction("Settings")
        self.menu_settings.triggered.connect(self.on_settings)
        self.menu.addAction(
            QIcon.fromTheme("application-exit"),
            "Exit",
            lambda: QCoreApplication.instance().quit(),
        )

    def _setup_app_menu(self, path: str, menu: QMenu):
        for app_id, manifest in self.host.app_store.list_app(path):
            if manifest is None:
                # Folder
                submenu = menu.addMenu(QIcon.fromTheme("folder"), app_id)
                self._setup_app_menu(app_id, submenu)
                continue

            app_menu = menu.addAction(manifest["appName"])
            app_menu.triggered.connect(
                lambda _=None, app_id=app_id: self.host.launch_app_id(app_id)
            )
            app_menu.setToolTip(manifest["description"])

            icon = self.host.app_store.icon(app_id)
            if icon:
                app_menu.setIcon(icon)

    @Slot()
    def update_menu(self):
        self._setup_menu()


def tray_icon() -> TrayIcon:
    return _tray_icon
