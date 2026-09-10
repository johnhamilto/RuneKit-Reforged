"""Stream the game window through ScreenCaptureKit and report what a live
frame source would give the clue screener: frame cadence, copy cost, dirty
rectangles, delivery thread, and how a stream frame compares with the
screenshot path the app uses today.

    python stream_check.py [--seconds 8] [--fps 10] [--native]

Run with the game open. Output size is logical points unless --native asks
for device pixels.
"""
import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np
import objc
import CoreMedia
import Foundation
import Quartz
import ScreenCaptureKit as SCK

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from runekit.game.quartz import instance as qi  # noqa: E402

BGRA = 0x42475241  # kCVPixelFormatType_32BGRA
READ_ONLY = 1  # kCVPixelBufferLock_ReadOnly


def pixels(pixel_buffer) -> np.ndarray:
    Quartz.CVPixelBufferLockBaseAddress(pixel_buffer, READ_ONLY)
    try:
        w = Quartz.CVPixelBufferGetWidth(pixel_buffer)
        h = Quartz.CVPixelBufferGetHeight(pixel_buffer)
        bpr = Quartz.CVPixelBufferGetBytesPerRow(pixel_buffer)
        base = Quartz.CVPixelBufferGetBaseAddress(pixel_buffer)
        raw = np.frombuffer(base.as_buffer(bpr * h), dtype=np.uint8)
        return raw.reshape(h, bpr // 4, 4)[:, :w, :].copy()
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(pixel_buffer, READ_ONLY)


class Sink(Foundation.NSObject, protocols=[objc.protocolNamed("SCStreamOutput"), objc.protocolNamed("SCStreamDelegate")]):
    def init(self):
        self = objc.super(Sink, self).init()
        self.lock = threading.Lock()
        self.frames = []  # (t, copy_ms, dirty_rects, shape)
        self.last = None
        self.idle = 0
        self.other = 0
        self.main_thread_hits = 0
        self.errors = []
        return self

    def stream_didOutputSampleBuffer_ofType_(self, stream, sample_buffer, kind):
        if kind != SCK.SCStreamOutputTypeScreen:
            return
        attachments = CoreMedia.CMSampleBufferGetSampleAttachmentsArray(sample_buffer, False)
        info = attachments[0] if attachments else {}
        status = info.get(SCK.SCStreamFrameInfoStatus)
        if status == SCK.SCFrameStatusIdle:
            self.idle += 1
            return
        if status != SCK.SCFrameStatusComplete:
            self.other += 1
            return
        pixel_buffer = CoreMedia.CMSampleBufferGetImageBuffer(sample_buffer)
        if pixel_buffer is None:
            self.other += 1
            return
        t0 = time.perf_counter()
        frame = pixels(pixel_buffer)
        copy_ms = (time.perf_counter() - t0) * 1000
        rects = []
        for entry in info.get(SCK.SCStreamFrameInfoDirtyRects) or []:
            ok, rect = Quartz.CGRectMakeWithDictionaryRepresentation(entry, None)
            if ok:
                rects.append((rect.origin.x, rect.origin.y, rect.size.width, rect.size.height))
        with self.lock:
            self.frames.append((time.perf_counter(), copy_ms, rects, frame.shape))
            self.last = frame
            if Foundation.NSThread.isMainThread():
                self.main_thread_hits += 1

    def stream_didStopWithError_(self, stream, error):
        self.errors.append(str(error))


def wait_for(flag, timeout):
    deadline = time.time() + timeout
    while not flag and time.time() < deadline:
        Quartz.CFRunLoopRunInMode(Quartz.kCFRunLoopDefaultMode, 0.01, True)
    return bool(flag)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=8)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--native", action="store_true", help="capture device pixels instead of points")
    args = ap.parse_args()

    windows = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID)
    game = next((w for w in windows if w.get(Quartz.kCGWindowOwnerName) == "rs2client"), None)
    if game is None:
        print("no rs2client window on screen")
        return 2
    wid = int(game[Quartz.kCGWindowNumber])
    bounds = game[Quartz.kCGWindowBounds]
    pw, ph = int(bounds["Width"]), int(bounds["Height"])

    content = qi._shareable_content()
    window = next(w for w in content.windows() if w.windowID() == wid)
    flt = SCK.SCContentFilter.alloc().initWithDesktopIndependentWindow_(window)
    scale = flt.pointPixelScale() or 1
    out_w, out_h = (pw * scale, ph * scale) if args.native else (pw, ph)

    config = SCK.SCStreamConfiguration.alloc().init()
    config.setWidth_(out_w)
    config.setHeight_(out_h)
    config.setPixelFormat_(BGRA)
    config.setShowsCursor_(False)
    config.setQueueDepth_(3)
    config.setMinimumFrameInterval_(CoreMedia.CMTimeMake(1, args.fps))

    sink = Sink.alloc().init()
    stream = SCK.SCStream.alloc().initWithFilter_configuration_delegate_(flt, config, sink)
    ok, err = stream.addStreamOutput_type_sampleHandlerQueue_error_(sink, SCK.SCStreamOutputTypeScreen, None, None)
    if not ok:
        print("addStreamOutput failed:", err)
        return 1

    started = []
    stream.startCaptureWithCompletionHandler_(lambda error: started.append(error))
    if not wait_for(started, 5):
        print("startCapture never completed")
        return 1
    if started[0] is not None:
        print("startCapture failed:", started[0])
        return 1
    t_start = time.perf_counter()
    print(f"streaming window {wid} at {out_w}x{out_h} ({'device px' if args.native else 'points'}), cap {args.fps} fps, for {args.seconds:.0f}s")

    while time.perf_counter() - t_start < args.seconds:
        Quartz.CFRunLoopRunInMode(Quartz.kCFRunLoopDefaultMode, 0.05, True)

    stopped = []
    stream.stopCaptureWithCompletionHandler_(lambda error: stopped.append(error))
    wait_for(stopped, 5)

    with sink.lock:
        frames = list(sink.frames)
        last = sink.last
    if not frames:
        print(f"no complete frames; idle={sink.idle} other={sink.other} errors={sink.errors}")
        return 1
    elapsed = frames[-1][0] - frames[0][0] if len(frames) > 1 else 0
    gaps = np.diff([f[0] for f in frames]) * 1000 if len(frames) > 1 else np.array([0.0])
    copies = np.array([f[1] for f in frames])
    shape = frames[0][3]
    area = shape[0] * shape[1]
    dirty_fracs = []
    top_band_hits = 0
    for _, _, rects, _ in frames:
        covered = sum(w * h for _, _, w, h in rects)
        dirty_fracs.append(min(1.0, covered / area) if area else 0)
        if any(y < shape[0] * 0.25 for _, y, _, _ in rects):
            top_band_hits += 1
    print(f"frames: {len(frames)} complete, {sink.idle} idle, {sink.other} other, in {elapsed:.1f}s "
          f"-> {len(frames) / elapsed if elapsed else 0:.1f} fps; gaps median {np.median(gaps):.0f} ms, max {gaps.max():.0f} ms")
    print(f"frame shape {shape}, copy median {np.median(copies):.1f} ms, max {copies.max():.1f} ms")
    print(f"dirty area median {np.median(dirty_fracs) * 100:.0f}% of frame, {top_band_hits}/{len(frames)} frames dirty in the top quarter")
    print(f"delivered on main thread: {sink.main_thread_hits}/{len(frames)}; stream errors: {sink.errors or 'none'}")

    # compare the last stream frame with a screenshot of the same window at the same size
    if not args.native:
        try:
            shot = qi._sck_screenshot(flt, pw, ph)
            data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(shot))
            h, w, bpr = Quartz.CGImageGetHeight(shot), Quartz.CGImageGetWidth(shot), Quartz.CGImageGetBytesPerRow(shot)
            shot_px = np.frombuffer(data, dtype=np.uint8)[: h * bpr].reshape(h, bpr // 4, 4)[:, :w, :]
            if shot_px.shape == last.shape:
                diff = np.abs(shot_px[..., :3].astype(int) - last[..., :3].astype(int)).max(axis=2)
                print(f"screenshot vs last stream frame: {np.count_nonzero(diff) / diff.size * 100:.1f}% pixels differ overall, "
                      f"{np.count_nonzero(diff[:40, -200:])} px in the top-right 200x40 (static interface bar)")
            else:
                print(f"screenshot shape {shot_px.shape} != stream shape {last.shape}")
        except Exception as e:  # the comparison is informational only
            print("screenshot comparison failed:", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
