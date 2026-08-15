import io
import logging
import os
import tempfile
import threading
import time
from typing import TYPE_CHECKING

import Quartz
import ScreenCaptureKit
import objc
from PIL import Image
from PySide6.QtCore import QRect
from PySide6.QtGui import QGuiApplication
import ApplicationServices
from PySide6.QtWidgets import QGraphicsItem

from ..instance import GameInstance
from ..psutil_mixins import PsUtilNetStat

if TYPE_CHECKING:
    from .manager import QuartzGameManager

_debug_dump_file = False
logger = logging.getLogger(__name__)


def cgrectref_to_qrect(cgrectref) -> QRect:
    _, cgrect = Quartz.CGRectMakeWithDictionaryRepresentation(cgrectref, None)
    return QRect(
        cgrect.origin.x, cgrect.origin.y, cgrect.size.width, cgrect.size.height
    )


def cgimageref_to_image(imgref) -> Image:
    buf = Quartz.CFDataCreateMutable(None, 0)

    dest = Quartz.CGImageDestinationCreateWithData(buf, "public.tiff", 1, None)
    Quartz.CGImageDestinationAddImage(dest, imgref, None)
    Quartz.CGImageDestinationFinalize(dest)

    buf_size = Quartz.CFDataGetLength(buf)
    py_buf = io.BytesIO()
    py_buf.write(Quartz.CFDataGetBytePtr(buf).as_buffer(buf_size))
    py_buf.seek(0)

    out = Image.open(py_buf, formats=("TIFF",))

    if _debug_dump_file:
        out.save(os.path.join(tempfile.gettempdir(), "game.bmp"))
        with open(os.path.join(tempfile.gettempdir(), "native.xbm"), "wb") as f:
            f.write(py_buf.getbuffer())

    return out


class ScreenCaptureError(Exception):
    pass


# CGWindowListCreateImage cannot capture windows on macOS 14+, so captures go
# through ScreenCaptureKit instead. Its APIs are async; _sck_call waits for the
# completion handler so callers keep the old synchronous interface.
_sck_lock = threading.Lock()
_sck_content = None


def _sck_call(func, *args):
    result = {}
    done = threading.Event()

    def completion(value, error):
        result["value"] = value
        result["error"] = error
        done.set()

    func(*args, completion)
    if not done.wait(5):
        raise ScreenCaptureError("ScreenCaptureKit request timed out")
    if result["error"] is not None or result["value"] is None:
        raise ScreenCaptureError(str(result["error"] or "no data returned"))
    return result["value"]


def _shareable_content(refresh=False):
    global _sck_content
    with _sck_lock:
        if _sck_content is None or refresh:
            # ScreenCaptureKit aborts the process if no window server
            # connection exists yet (CGS_REQUIRE_INIT)
            Quartz.CGMainDisplayID()
            _sck_content = _sck_call(
                ScreenCaptureKit.SCShareableContent.getShareableContentWithCompletionHandler_
            )
        return _sck_content


def _sck_screenshot(content_filter, width, height, source_rect=None):
    config = ScreenCaptureKit.SCStreamConfiguration.alloc().init()
    config.setWidth_(int(width))
    config.setHeight_(int(height))
    config.setShowsCursor_(False)
    config.setIgnoreShadowsSingleWindow_(True)
    if source_rect is not None:
        config.setSourceRect_(source_rect)
    return _sck_call(
        ScreenCaptureKit.SCScreenshotManager.captureImageWithFilter_configuration_completionHandler_,
        content_filter,
        config,
    )


def sck_capture_window(wid, width, height, refresh=False):
    content = _shareable_content(refresh)
    window = next((w for w in content.windows() if w.windowID() == wid), None)
    if window is None:
        if refresh:
            raise ScreenCaptureError(f"Window {wid} not found by ScreenCaptureKit")
        return sck_capture_window(wid, width, height, refresh=True)

    content_filter = (
        ScreenCaptureKit.SCContentFilter.alloc().initWithDesktopIndependentWindow_(
            window
        )
    )
    return _sck_screenshot(content_filter, width, height)


def sck_capture_desktop(x, y, w, h):
    content = _shareable_content()
    display = next(
        (
            d
            for d in content.displays()
            if Quartz.CGRectContainsPoint(d.frame(), Quartz.CGPointMake(x, y))
        ),
        content.displays()[0],
    )
    content_filter = (
        ScreenCaptureKit.SCContentFilter.alloc().initWithDisplay_excludingWindows_(
            display, []
        )
    )
    scale = content_filter.pointPixelScale() or 1
    frame = display.frame()
    rect = Quartz.CGRectMake(x - frame.origin.x, y - frame.origin.y, w, h)
    return _sck_screenshot(content_filter, w * scale, h * scale, source_rect=rect)


# This decorator does not support being a class method
@objc.callbackFor(ApplicationServices.AXObserverCreate)
def on_ax_event(observer, element, notification, ptr):
    try:
        self: "QuartzGameInstance" = objc.context.get(ptr)
    except KeyError:
        logger.warning(
            "Received AX event callback for missing pointer %d, removing", ptr
        )
        ApplicationServices.AXObserverRemoveNotification(
            observer, element, notification
        )
        return

    if notification == ApplicationServices.kAXApplicationActivatedNotification:
        self._is_active = True
        self.focusChanged.emit(True)
    elif notification == ApplicationServices.kAXApplicationDeactivatedNotification:
        self._is_active = False
        self.focusChanged.emit(False)
    elif notification == ApplicationServices.kAXWindowResizedNotification:
        self.positionChanged.emit(self.get_position())
    elif notification == ApplicationServices.kAXWindowMovedNotification:
        self.positionChanged.emit(self.get_position())
    else:
        self.logger.warning("Got unknown AX event %s", notification)


class QuartzGameInstance(PsUtilNetStat, GameInstance):
    _is_active = False
    overlay: QGraphicsItem

    __game_last_grab = 0.0
    __game_last_image = None

    def __init__(self, manager: "QuartzGameManager", wid, pid, **kwargs):
        super().__init__(**kwargs)
        self.manager = manager
        self.wid = wid
        self.pid = pid
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}:{pid}")
        self.obj_pointer = objc.context.register(self)

        self._setup_observer()
        self.update_is_active()
        self.overlay, self._overlay_disconnect = self.manager.overlay.add_instance(self)

    def _setup_observer(self):
        self._ax_element = ApplicationServices.AXUIElementCreateApplication(self.pid)
        self._ax_observed = [
            ApplicationServices.kAXApplicationActivatedNotification,
            ApplicationServices.kAXApplicationDeactivatedNotification,
            ApplicationServices.kAXWindowResizedNotification,
            ApplicationServices.kAXWindowMovedNotification,
        ]
        err, self._observer = ApplicationServices.AXObserverCreate(
            self.pid, on_ax_event, None
        )
        if err != ApplicationServices.kAXErrorSuccess:
            raise AXAPIError(err)

        for item in self._ax_observed:
            err = ApplicationServices.AXObserverAddNotification(
                self._observer, self._ax_element, item, self.obj_pointer
            )
            if err != ApplicationServices.kAXErrorSuccess:
                raise AXAPIError(err)

        Quartz.CFRunLoopAddSource(
            Quartz.CFRunLoopGetCurrent(),
            ApplicationServices.AXObserverGetRunLoopSource(self._observer),
            Quartz.kCFRunLoopCommonModes,
        )

    def __del__(self):
        self.logger.debug("Destructor")
        for item in self._ax_observed:
            ApplicationServices.AXObserverRemoveNotification(
                self._observer, self._ax_element, item
            )

        objc.context.unregister(self)
        self._overlay_disconnect()

    def get_position(self) -> QRect:
        # The docs say this API is expensive...
        infos = Quartz.CGWindowListCreateDescriptionFromArray([self.wid])
        info = infos[0]  # FIXME: what if window closed
        return cgrectref_to_qrect(info[Quartz.kCGWindowBounds])

    def get_scaling(self) -> float:
        screen = QGuiApplication.screenAt(self.get_position().topLeft())
        return screen.devicePixelRatio()

    def is_focused(self) -> bool:
        return self._is_active

    def update_is_active(self):
        self._is_active = (
            Quartz.NSWorkspace.sharedWorkspace()
            .frontmostApplication()
            .processIdentifier()
            == self.pid
        )

    def grab_game(self) -> Image:
        # FIXME: Crop title bar
        if (time.monotonic() - self.__game_last_grab) * 1000 < self.refresh_rate:
            return self.__game_last_image

        bounds = self.get_position()
        scale = self.get_scaling()
        width = int(bounds.width() * scale)
        height = int(bounds.height() * scale)

        try:
            imgref = sck_capture_window(self.wid, width, height)
        except ScreenCaptureError:
            try:
                # the cached window reference may be stale, retry with a fresh one
                imgref = sck_capture_window(self.wid, width, height, refresh=True)
            except ScreenCaptureError:
                if not Quartz.CGPreflightScreenCaptureAccess():
                    self.manager.request_accessibility_popup.emit()
                    raise NoCapturePermission
                raise

        out = cgimageref_to_image(imgref)
        if scale > 1:
            out = out.resize(
                (int(out.width / scale), int(out.height / scale)), Image.NEAREST
            )

        self.__game_last_grab = time.monotonic()
        self.__game_last_image = out
        return out

    def grab_desktop(self, x: int, y: int, w: int, h: int) -> Image:
        imgref = sck_capture_desktop(x, y, w, h)
        out = cgimageref_to_image(imgref)
        return out.resize((w, h), Image.NEAREST)

    def get_overlay_area(self) -> QGraphicsItem:
        return self.overlay


class AXAPIError(Exception):
    mapping = {
        ApplicationServices.kAXErrorInvalidUIElementObserver: "The observer is not a valid AXObserverRef type",
        ApplicationServices.kAXErrorIllegalArgument: "One or more of the arguments is an illegal value or the length of the notification name is greater than 1024",
        ApplicationServices.kAXErrorNotificationUnsupported: "The observer is not a valid AXObserverRef type",
        ApplicationServices.kAXErrorNotificationAlreadyRegistered: "The notification has already been registered",
        ApplicationServices.kAXErrorCannotComplete: "The function cannot complete because messaging has failed in some way.",
        ApplicationServices.kAXErrorFailure: "There is some sort of system memory failure.",
        ApplicationServices.kAXErrorAPIDisabled: "Assistive applications are not enabled in System Preferences.",
    }

    def __init__(self, code):
        if code == 0:
            raise ValueError("Success")

        super().__init__(self.mapping.get(code, f"API Error: {code}"))


class NoCapturePermission(Exception):
    def __init__(self):
        super().__init__("Screen Recording permission is not allowed")
