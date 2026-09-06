import logging
import time
from functools import reduce
from typing import List, Dict, Optional, Union

import Quartz
import ApplicationServices
from PySide6.QtCore import QTimer, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QMessageBox

from .instance import QuartzGameInstance
from runekit.game.overlay import DesktopWideOverlay
from ..instance import GameInstance
from ..manager import GameManager

has_prompted_accessibility = False
logger = logging.getLogger(__name__)


class QuartzGameManager(GameManager):
    _instances: Dict[int, GameInstance]
    overlay: DesktopWideOverlay

    request_accessibility_popup = Signal()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._instances = {}
        self._alt_down = False
        self._mouse_tap = None
        self._mouse_capture = False
        self._mouse_held = False  # a swallowed press is waiting for its release
        self.request_accessibility_popup.connect(self.accessibility_popup)
        self._setup_tap()

        ApplicationServices.AXIsProcessTrustedWithOptions(
            {
                ApplicationServices.kAXTrustedCheckOptionPrompt: True,
            }
        )
        while not ApplicationServices.AXIsProcessTrusted():
            time.sleep(0.1)

        self._setup_overlay()

    def _setup_tap(self):
        events = [
            Quartz.kCGEventLeftMouseDown,
            Quartz.kCGEventRightMouseDown,
            Quartz.kCGEventKeyDown,
            Quartz.kCGEventFlagsChanged,
        ]
        events = [Quartz.CGEventMaskBit(e) for e in events]
        event_mask = reduce(lambda a, b: a | b, events)
        self._tap = Quartz.CGEventTapCreate(
            Quartz.kCGAnnotatedSessionEventTap,
            Quartz.kCGTailAppendEventTap,
            Quartz.kCGEventTapOptionListenOnly,  # TODO: Tap keydown synchronously
            event_mask,
            self._on_input,
            None,
        )
        source = Quartz.CFMachPortCreateRunLoopSource(None, self._tap, 0)
        Quartz.CFRunLoopAddSource(
            Quartz.CFRunLoopGetCurrent(), source, Quartz.kCFRunLoopCommonModes
        )

    def _setup_overlay(self):
        self.overlay = DesktopWideOverlay()

        def start():
            self.overlay.show()
            self.overlay.check_compatibility()

        # Seems like QGraphicsView has a delay before applying stylesheet
        # Put some delay to allow it to initialize and not flash
        QTimer.singleShot(1000, start)

    def stop(self):
        try:
            self.overlay.hide()
            self.overlay.deleteLater()
        except RuntimeError:
            pass

    def get_instances(self) -> List[GameInstance]:
        windows = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID
        )

        for window in windows:
            if window[Quartz.kCGWindowOwnerName] == "rs2client":
                wid = int(window[Quartz.kCGWindowNumber])
                if wid not in self._instances:
                    pid = int(window[Quartz.kCGWindowOwnerPID])
                    self._instances[wid] = QuartzGameInstance(
                        self, wid, pid, parent=self
                    )

        return list(self._instances.values())

    def get_active_instance(self) -> Union[GameInstance, None]:
        if not self._instances:
            return None

        return list(self._instances.values())[0]

    def get_instance_by_pid(self, pid: int) -> Optional[QuartzGameInstance]:
        for instance in self._instances.values():
            if instance.pid == pid:
                return instance

    def _on_input(self, proxy, type_, event, _):
        event_type = Quartz.CGEventGetType(event)
        if event_type == Quartz.kCGEventTapDisabledByUserInput:
            QTimer.singleShot(0, self.accessibility_popup)
            return event
        elif event_type == Quartz.kCGEventTapDisabledByTimeout:
            Quartz.CGEventTapEnable(self._tap, True)
            return event

        nsevent = Quartz.NSEvent.eventWithCGEvent_(event)
        if nsevent.type() == Quartz.NSEventTypeFlagsChanged:
            self._on_flags_changed(nsevent)
            return event
        if nsevent.type() == Quartz.NSEventTypeKeyDown:
            front_app = Quartz.NSWorkspace.sharedWorkspace().frontmostApplication()
            instance = self.get_instance_by_pid(front_app.processIdentifier())
        else:
            instance = self._instances.get(nsevent.windowNumber())

        if not instance:
            return event

        # Check for cmd1
        if nsevent.type() == Quartz.NSEventTypeKeyDown:
            if (
                nsevent.keyCode() == 18
                and nsevent.modifierFlags() & Quartz.NSEventModifierFlagCommand
            ):
                instance.alt1_pressed.emit()
                return None

        instance.game_activity.emit()

        return event

    def _on_flags_changed(self, nsevent):
        alt = bool(nsevent.modifierFlags() & Quartz.NSEventModifierFlagOption)
        if alt == self._alt_down:
            return
        if alt:
            front_app = Quartz.NSWorkspace.sharedWorkspace().frontmostApplication()
            if not self.get_instance_by_pid(front_app.processIdentifier()):
                return
        self._alt_down = alt
        # leave the tap callback before any window work happens
        QTimer.singleShot(0, lambda: self.alt_changed.emit(alt))

    def set_mouse_capture(self, on: bool):
        self._mouse_capture = on
        if self._mouse_tap is None:
            if not on:
                return
            events = [
                Quartz.kCGEventLeftMouseDown,
                Quartz.kCGEventLeftMouseDragged,
                Quartz.kCGEventLeftMouseUp,
            ]
            mask = reduce(lambda a, b: a | b, [Quartz.CGEventMaskBit(e) for e in events])
            self._mouse_tap = Quartz.CGEventTapCreate(
                Quartz.kCGAnnotatedSessionEventTap,
                Quartz.kCGHeadInsertEventTap,
                Quartz.kCGEventTapOptionDefault,
                mask,
                self._on_mouse,
                None,
            )
            if self._mouse_tap is None:
                logger.warning("Could not create the mouse event tap; marker editing is off")
                return
            source = Quartz.CFMachPortCreateRunLoopSource(None, self._mouse_tap, 0)
            Quartz.CFRunLoopAddSource(
                Quartz.CFRunLoopGetCurrent(), source, Quartz.kCFRunLoopCommonModes
            )
            return  # a new tap starts enabled
        if on or not self._mouse_held:
            Quartz.CGEventTapEnable(self._mouse_tap, on)

    def _on_mouse(self, proxy, type_, event, _):
        if type_ in (Quartz.kCGEventTapDisabledByTimeout, Quartz.kCGEventTapDisabledByUserInput):
            Quartz.CGEventTapEnable(self._mouse_tap, True)
            return event
        hook = self.mouse_hook
        if hook is None:
            return event
        if type_ == Quartz.kCGEventLeftMouseDown:
            if not self._mouse_capture:
                return event
            kind = "down"
        elif not self._mouse_held:
            return event  # the press went to the game, so does the rest
        else:
            kind = "drag" if type_ == Quartz.kCGEventLeftMouseDragged else "up"
        point = Quartz.CGEventGetLocation(event)
        try:
            consumed = bool(hook(kind, point.x, point.y))
        except Exception:
            logger.exception("Mouse hook failed")
            consumed = False
        if kind == "down":
            self._mouse_held = consumed
        elif kind == "up":
            self._mouse_held = False
            if not self._mouse_capture:
                Quartz.CGEventTapEnable(self._mouse_tap, False)
        return None if consumed else event

    @Slot()
    def accessibility_popup(self):
        global has_prompted_accessibility
        if has_prompted_accessibility:
            return

        has_prompted_accessibility = True
        msgbox = QMessageBox(
            QMessageBox.Icon.Warning,
            "Permission required",
            "RuneKit needs Screen Recording permission\n\nOpen System Preferences > Security > Privacy > Screen Recording to allow this",
            QMessageBox.StandardButton.Open | QMessageBox.StandardButton.Ignore,
        )
        button = msgbox.exec()

        if button == QMessageBox.StandardButton.Open:
            QDesktopServices.openUrl(
                "x-apple.systempreferences:com.apple.preference.security?Privacy_Screen Recording"
            )
