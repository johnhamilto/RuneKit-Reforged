"""A live ScreenCaptureKit stream of one window, kept at logical size so the
latest frame can stand in for a screenshot. Frames only arrive when the
window content changes, so the last complete frame stays current; a frame
copy costs a few milliseconds on the stream's own thread."""
import logging
import threading
import time
from typing import Optional

import CoreMedia
import Foundation
import objc
import Quartz
import ScreenCaptureKit
from PIL import Image

logger = logging.getLogger(__name__)

BGRA = 0x42475241  # kCVPixelFormatType_32BGRA
READ_ONLY = 1  # kCVPixelBufferLock_ReadOnly
STALL_S = 3.0  # no frame or idle notice for this long: fall back to screenshots


class _Sink(
    Foundation.NSObject,
    protocols=[objc.protocolNamed("SCStreamOutput"), objc.protocolNamed("SCStreamDelegate")],
):
    def init(self):
        self = objc.super(_Sink, self).init()
        if self is None:
            return None
        self.lock = threading.Lock()
        self.frame = None  # (monotonic time, PIL RGBA image)
        self.last_activity = time.monotonic()
        self.error = None
        return self

    def stream_didOutputSampleBuffer_ofType_(self, stream, sample_buffer, kind):
        if kind != ScreenCaptureKit.SCStreamOutputTypeScreen:
            return
        attachments = CoreMedia.CMSampleBufferGetSampleAttachmentsArray(sample_buffer, False)
        info = attachments[0] if attachments else {}
        now = time.monotonic()
        if info.get(ScreenCaptureKit.SCStreamFrameInfoStatus) != ScreenCaptureKit.SCFrameStatusComplete:
            with self.lock:
                self.last_activity = now  # idle notice: nothing changed, the last frame holds
            return
        pixel_buffer = CoreMedia.CMSampleBufferGetImageBuffer(sample_buffer)
        if pixel_buffer is None:
            return
        Quartz.CVPixelBufferLockBaseAddress(pixel_buffer, READ_ONLY)
        try:
            width = Quartz.CVPixelBufferGetWidth(pixel_buffer)
            height = Quartz.CVPixelBufferGetHeight(pixel_buffer)
            stride = Quartz.CVPixelBufferGetBytesPerRow(pixel_buffer)
            base = Quartz.CVPixelBufferGetBaseAddress(pixel_buffer)
            # BGRA -> RGBA decodes into a fresh image, so the buffer can be unlocked after
            image = Image.frombuffer(
                "RGBA", (width, height), memoryview(base.as_buffer(stride * height)), "raw", "BGRA", stride, 1
            )
        finally:
            Quartz.CVPixelBufferUnlockBaseAddress(pixel_buffer, READ_ONLY)
        with self.lock:
            self.frame = (now, image)
            self.last_activity = now

    def stream_didStopWithError_(self, stream, error):
        with self.lock:
            self.error = str(error) if error is not None else "stopped"
        logger.warning("Window stream stopped: %s", self.error)


def _configuration(width: int, height: int, fps: int):
    config = ScreenCaptureKit.SCStreamConfiguration.alloc().init()
    config.setWidth_(int(width))
    config.setHeight_(int(height))
    config.setPixelFormat_(BGRA)
    config.setShowsCursor_(False)
    config.setQueueDepth_(3)
    config.setMinimumFrameInterval_(CoreMedia.CMTimeMake(1, fps))
    return config


class WindowStream:
    def __init__(self, content_filter, width: int, height: int, fps: int = 10):
        self.width = int(width)
        self.height = int(height)
        self.fps = fps
        self._filter = content_filter
        self._sink = _Sink.alloc().init()
        self._stream = None

    def start(self):
        """Begin capturing; raises RuntimeError if the output cannot be attached.
        Start failures reported later show up as failed()."""
        stream = ScreenCaptureKit.SCStream.alloc().initWithFilter_configuration_delegate_(
            self._filter, _configuration(self.width, self.height, self.fps), self._sink
        )
        ok, error = stream.addStreamOutput_type_sampleHandlerQueue_error_(
            self._sink, ScreenCaptureKit.SCStreamOutputTypeScreen, None, None
        )
        if not ok:
            raise RuntimeError(f"could not attach stream output: {error}")
        self._stream = stream
        self._sink.last_activity = time.monotonic()
        stream.startCaptureWithCompletionHandler_(self._on_started)

    def _on_started(self, error):
        if error is not None:
            with self._sink.lock:
                self._sink.error = str(error)
            logger.warning("Window stream failed to start: %s", error)
        else:
            logger.info("Window stream started at %dx%d, %d fps", self.width, self.height, self.fps)

    def stop(self):
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stopCaptureWithCompletionHandler_(lambda error: None)
            logger.info("Window stream stopped")

    def resize(self, width: int, height: int):
        width, height = int(width), int(height)
        if (width, height) == (self.width, self.height) or self._stream is None:
            return
        self.width, self.height = width, height
        self._stream.updateConfiguration_completionHandler_(
            _configuration(width, height, self.fps),
            lambda error: error is not None and logger.warning("Window stream resize failed: %s", error),
        )

    def failed(self) -> bool:
        with self._sink.lock:
            return self._stream is None or self._sink.error is not None

    def latest(self) -> Optional[Image.Image]:
        """The current frame, or None when there is none yet or the stream stalled."""
        with self._sink.lock:
            if self._stream is None or self._sink.error is not None or self._sink.frame is None:
                return None
            if time.monotonic() - self._sink.last_activity >= STALL_S:
                return None
            return self._sink.frame[1]
