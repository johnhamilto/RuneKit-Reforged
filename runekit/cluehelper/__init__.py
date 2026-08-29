"""Built-in clue solver for treasure trail clues."""
import html
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image
from PySide6.QtCore import QObject, QSettings, QStandardPaths, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QFont, QPen
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsSimpleTextItem

from runekit import detection
from runekit.cluehelper import compass as compass_mod
from runekit.cluehelper import lockbox as lockbox_mod
from runekit.cluehelper import maps as maps_mod
from runekit.cluehelper import slide as slide_mod
from runekit.cluehelper import slide_solver, solver, vision
from runekit.cluehelper import towers as towers_mod
from runekit.cluehelper import wiki as wiki_mod
from runekit.cluehelper import window as window_mod
from runekit.cluehelper import worldmap
from runekit.cluehelper.assets import ClueAssets

if TYPE_CHECKING:
    from runekit.game import GameInstance

logger = logging.getLogger(__name__)

OVERLAY_MOVES = 25
OVERLAY_TIMEOUT_MS = 20000

AUTO_INTERVAL_MS = 4000
AUTO_COOLDOWN_S = 10
AUTO_REARM_MISSES = 2
SCREEN_TITLES = ("mysterious clue scroll", "treasure map", "lockbox", "towers", "celtic knot")
SCREEN_WIDTH = 1700  # frames are shrunk to this for fast screening


def _load_hint(cache_dir: Path):
    try:
        return json.loads((cache_dir / "scale_hint.json").read_text())["scale"]
    except Exception:
        return None


class _ScanThread(QThread):
    """Cheap screen check for a clue-like interface: fast OCR for known
    titles at reduced resolution, plus a slide needle probe once a scale
    hint exists."""

    verdict = Signal(bool)

    def __init__(self, cache_dir: Path, parent=None):
        super().__init__(parent=parent)
        self.instance = None
        self.cache_dir = cache_dir
        self._runs = 0

    def run(self):
        try:
            self._runs += 1
            frame = solver._to_image(self.instance.grab_game())
            if frame.width > SCREEN_WIDTH:
                frame_small = frame.resize(
                    (SCREEN_WIDTH, round(frame.height * SCREEN_WIDTH / frame.width)),
                    Image.BILINEAR,
                )
            else:
                frame_small = frame
            for line in vision.ocr_lines(frame_small, fast=True):
                if max(solver._ratio(line["text"], t) for t in SCREEN_TITLES) >= 0.6:
                    self.verdict.emit(True)
                    return
            # slide puzzles have no title; probe for the interface sprite.
            # Cheap with a scale hint, so do the full sweep only occasionally
            # until one is known.
            hint = _load_hint(self.cache_dir)
            if hint or self._runs % 8 == 1:
                needle = detection.to_array(ClueAssets(self.cache_dir).needle("slide"))
                m = detection.calibrate_scale(detection.to_array(frame), needle, hint=hint)
                if m.ok:
                    self.verdict.emit(True)
                    return
            self.verdict.emit(False)
        except Exception:
            logger.debug("Clue screening failed", exc_info=True)
            self.verdict.emit(False)


class _SolveThread(QThread):
    ok = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, instance: "GameInstance", cache_dir: Path, parent=None):
        super().__init__(parent=parent)
        self.instance = instance
        self.cache_dir = cache_dir

    def _load_hint(self):
        return _load_hint(self.cache_dir)

    def _save_hint(self, scale: float):
        try:
            (self.cache_dir / "scale_hint.json").write_text(json.dumps({"scale": scale}))
        except Exception:
            pass

    def _puzzle_title(self, result) -> str:
        for line in result.lines:
            for title in ("lockbox", "towers", "celtic knot"):
                if solver._ratio(line["text"], title) >= 0.75:
                    return title
        return ""

    def _attach_map(self, result):
        if not result.matches:
            return
        try:
            spots, level, primary = solver.entry_spots(result.matches[0][1])
            if spots:
                self.progress.emit("Fetching the world map…")
                result.map_image = worldmap.location_image(
                    self.cache_dir, spots, level, primary=primary
                )
        except Exception:
            logger.warning("World map snapshot failed", exc_info=True)

    def run(self):
        try:
            hint = self._load_hint()
            self.progress.emit("Reading screen text…")
            frame = self.instance.grab_game()
            dbs = solver.load_databases(self.cache_dir)
            result = solver.solve_frame(frame, dbs)
            if result.status == "unsupported":
                # a clue interface with no matchable text: try image matching
                self.progress.emit("Matching map image…")
                match = maps_mod.read_map_clue(detection.to_array(frame), dbs, ClueAssets(self.cache_dir), hint=hint)
                if match is not None:
                    if match.scale:
                        self._save_hint(match.scale)
                    entry = match.entry
                    if entry.get("x") is not None:
                        wiki_entry = wiki_mod.nearest(
                            dbs.get("wiki") or [], entry["x"], entry["z"], types=("map",)
                        )
                        if wiki_entry is not None:
                            entry = wiki_entry
                    result.status = "solved"
                    result.read_text = "(map image)"
                    result.matches = [(1.0, entry)]
                self._attach_map(result)
                self.ok.emit(result)
                return
            if result.status != "no_clue":
                self._attach_map(result)
                self.ok.emit(result)
                return

            arr = detection.to_array(frame)
            assets = ClueAssets(self.cache_dir)
            title = self._puzzle_title(result)

            if title == "lockbox":
                self.progress.emit("Reading the lockbox…")
                read = lockbox_mod.read_lockbox(arr, assets, hint=hint)
                if read is None:
                    self.failed.emit(
                        "Lockbox detected but the grid could not be read. "
                        "Make sure the whole box is visible and try again."
                    )
                    return
                self._save_hint(read.scale)
                sol = lockbox_mod.solve_lockbox(read.grid)
                if sol is None:
                    self.failed.emit(
                        "Lockbox grid read but no solution exists; likely a misread. Try again."
                    )
                    return
                self.ok.emit(lockbox_mod.LockboxSolution(read, sol[0], sol[1]))
                return

            if title == "towers":
                self.progress.emit("Reading the towers board…")
                read = towers_mod.read_towers(arr, assets, hint=hint)
                if read is None:
                    self.failed.emit(
                        "Towers puzzle detected but the clues could not be read. "
                        "Make sure the whole board is visible and try again."
                    )
                    return
                self._save_hint(read.scale)
                sols = towers_mod.solve_towers(read, limit=2)
                if not sols:
                    self.failed.emit(
                        "Towers clues read but no solution exists; likely a misread. Try again."
                    )
                    return
                self.ok.emit(towers_mod.TowersSolution(read, sols[0], len(sols)))
                return

            if title == "celtic knot":
                self.failed.emit("Found a celtic knot puzzle; that type isn't supported yet.")
                return

            self.progress.emit("Looking for a slide puzzle…")
            board = slide_mod.read_slide(arr, assets, hint=hint, debug_dir=self.cache_dir)
            if board is not None:
                self._save_hint(board.scale)
                try:
                    moves = slide_solver.solve(board.board)
                except ValueError as e:
                    self.failed.emit(
                        f"Slide puzzle detected but the board read is inconsistent ({e}). "
                        "Make sure no tile is mid-animation and try again."
                    )
                    return
                self.ok.emit(slide_mod.SlideSolution(board, moves))
                return

            self.progress.emit("Looking for a compass…")
            comp = compass_mod.read_compass(arr, assets, hint=hint)
            if comp is not None:
                if comp.scale:
                    self._save_hint(comp.scale)
                self.ok.emit(comp)
                return

            self.ok.emit(result)
        except Exception as e:
            logger.error("Clue solve failed", exc_info=True)
            self.failed.emit(str(e))


def _ocr_details(result) -> str:
    lines = [
        f"{html.escape(l['text'])} ({l['confidence']:.2f})"
        for l in result.lines
    ]
    if not lines:
        return ""
    return "Text found on screen:<br>" + "<br>".join(lines)


class ClueHelper(QObject):
    solve_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.cache_dir = Path(
            QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation)
        )
        self._thread = None
        self._instance = None
        self._window = None
        self.instance_provider = None  # set by the host; returns a GameInstance
        self._auto_solving = False
        self._auto_armed = True
        self._auto_misses = 0
        self._auto_last = 0.0
        self._scan_thread = None
        self._auto_timer = QTimer(self)
        self._auto_timer.setInterval(AUTO_INTERVAL_MS)
        self._auto_timer.timeout.connect(self._auto_tick)
        if QSettings().value("cluehelper/auto_detect", False, bool):
            self._auto_timer.start()

    @property
    def window(self) -> "window_mod.ClueSolverWindow":
        if self._window is None:
            self._window = window_mod.ClueSolverWindow()
            self._window.solve_requested.connect(self.solve_requested)
            self._window.auto_check.setChecked(self._auto_timer.isActive())
            self._window.auto_toggled.connect(self.set_auto_detect)
        return self._window

    def show_error(self, message: str):
        self.window.open()
        self.window.show_message(message)

    @Slot(bool)
    def set_auto_detect(self, on: bool):
        QSettings().setValue("cluehelper/auto_detect", on)
        if on and not self._auto_timer.isActive():
            self._auto_armed = True
            self._auto_timer.start()
        elif not on:
            self._auto_timer.stop()

    def _auto_tick(self):
        if self._thread is not None and self._thread.isRunning():
            return
        if self._scan_thread is not None and self._scan_thread.isRunning():
            return
        instance = self.instance_provider() if self.instance_provider else None
        if instance is None:
            return
        if self._scan_thread is None:
            self._scan_thread = _ScanThread(self.cache_dir, parent=self)
            self._scan_thread.verdict.connect(self._on_scan_verdict)
        self._scan_thread.instance = instance
        self._scan_thread.start()

    @Slot(bool)
    def _on_scan_verdict(self, found: bool):
        if not found:
            self._auto_misses += 1
            if self._auto_misses >= AUTO_REARM_MISSES:
                self._auto_armed = True
            return
        self._auto_misses = 0
        if not self._auto_armed or time.time() - self._auto_last < AUTO_COOLDOWN_S:
            return
        instance = self.instance_provider() if self.instance_provider else None
        if instance is None:
            return
        self._auto_armed = False
        self._auto_last = time.time()
        self.solve(instance, auto=True)

    @Slot()
    def solve(self, instance: "GameInstance", auto: bool = False):
        if auto:
            # update in place without stealing focus from the game
            self.window.show()
            self.window.raise_()
        else:
            self.window.open()
        if self._thread is not None and self._thread.isRunning():
            return

        self._auto_solving = auto
        self._instance = instance
        self.window.set_busy(True, "Capturing the game…")
        self._thread = _SolveThread(instance, self.cache_dir, parent=self)
        self._thread.ok.connect(self.on_result)
        self._thread.failed.connect(self.on_failed)
        self._thread.progress.connect(self.on_progress)
        self._thread.start()

    @Slot(str)
    def on_progress(self, text: str):
        self.window.set_busy(True, text)

    # ------------------------------------------------------------- overlays

    def _overlay_area(self):
        try:
            area = self._instance.get_overlay_area()
        except Exception:
            return None
        return area

    def _expire(self, items):
        def cleanup():
            for item in items:
                scene = item.scene()
                if scene is not None:
                    scene.removeItem(item)

        QTimer.singleShot(OVERLAY_TIMEOUT_MS, cleanup)

    def _draw_slide_moves(self, sol: "slide_mod.SlideSolution") -> bool:
        area = self._overlay_area()
        if area is None:
            return False

        bx, by, bw, _ = sol.board.board_rect
        tile = bw / 5
        items = []
        visits = {}
        for i, cell in enumerate(sol.moves[:OVERLAY_MOVES]):
            r, c = divmod(cell, 5)
            n = visits.get(cell, 0)
            visits[cell] = n + 1
            cx = bx + (c + 0.5) * tile + (n % 3 - 1) * 14
            cy = by + (r + 0.5) * tile + (n // 3 % 3 - 1) * 14
            dot = QGraphicsEllipseItem(cx - 10, cy - 10, 20, 20)
            dot.setBrush(QBrush(QColor(20, 20, 30, 180)))
            dot.setPen(QPen(QColor(255, 200, 40), 1.5))
            dot.setParentItem(area)
            label = QGraphicsSimpleTextItem(str(i + 1))
            label.setBrush(QBrush(QColor(255, 220, 80)))
            label.setFont(QFont("Verdana", 9, QFont.Weight.Bold))
            rect = label.boundingRect()
            label.setPos(cx - rect.width() / 2, cy - rect.height() / 2)
            label.setParentItem(area)
            items.extend((dot, label))

        self._expire(items)
        return True

    def _draw_grid_numbers(self, origin, stride, values, color=QColor(255, 220, 80)) -> bool:
        area = self._overlay_area()
        if area is None:
            return False

        ox, oy = origin
        items = []
        for r in range(5):
            for c in range(5):
                value = values[r][c]
                if not value:
                    continue
                cx = ox + (c + 0.5) * stride
                cy = oy + (r + 0.5) * stride
                label = QGraphicsSimpleTextItem(str(value))
                label.setBrush(QBrush(color))
                label.setPen(QPen(QColor(0, 0, 0), 0.5))
                label.setFont(QFont("Verdana", 14, QFont.Weight.Bold))
                rect = label.boundingRect()
                label.setPos(cx - rect.width() / 2, cy - rect.height() / 2)
                label.setParentItem(area)
                items.append(label)

        self._expire(items)
        return True

    # -------------------------------------------------------------- results

    def _show_slide(self, sol: "slide_mod.SlideSolution"):
        drew = self._draw_slide_moves(sol)
        pixmap = None
        try:
            theme = ClueAssets(self.cache_dir).slide_theme(sol.board.theme)
            count = min(OVERLAY_MOVES, len(sol.moves))
            pixmap = window_mod.render_slide(sol.board.board, sol.moves, theme, count)
        except Exception:
            logger.warning("Slide render failed", exc_info=True)
        body = (
            f"Solution: <b>{len(sol.moves)} moves</b>, the first "
            f"{min(OVERLAY_MOVES, len(sol.moves))} numbered"
            + (" here and on the game." if drew else " here.")
            + " Solve again for the next batch."
        )
        self.window.show_result(
            f"Slide puzzle read (confidence {sol.board.confidence:.0%}).",
            body,
            pixmap,
        )

    def _show_lockbox(self, sol: "lockbox_mod.LockboxSolution"):
        stride = lockbox_mod.TILE * sol.read.scale
        drew = self._draw_grid_numbers(sol.read.origin, stride, sol.presses)
        pixmap = None
        try:
            assets = ClueAssets(self.cache_dir)
            needles = [assets.needle(n) for n in lockbox_mod.TILE_NAMES]
            pixmap = window_mod.render_lockbox(sol.read.grid, sol.presses, needles)
        except Exception:
            logger.warning("Lockbox render failed", exc_info=True)
        total = sum(map(sum, sol.presses))
        body = (
            f"Press each tile the number of times shown"
            + (" (also drawn on the game)" if drew else "")
            + f": <b>{total} presses</b> makes every tile "
            f"{lockbox_mod.TILE_NAMES[sol.target]}."
        )
        self.window.show_result(
            f"Lockbox read (confidence {sol.read.confidence:.0%}).", body, pixmap
        )

    def _show_towers(self, sol: "towers_mod.TowersSolution"):
        s = sol.read.scale
        ix, iy = sol.read.inner_origin
        drew = self._draw_grid_numbers(
            (ix + 27 * s, iy + 27.5 * s), 44 * s, sol.grid, color=QColor(120, 235, 120)
        )
        pixmap = window_mod.render_towers(
            sol.grid, sol.read.filled, sol.read.top, sol.read.bot, sol.read.left, sol.read.right
        )
        body = "Fill the green numbers." + (" Also drawn on the game." if drew else "")
        if sol.solutions > 1:
            body += (
                "<br><b>Note:</b> the visible clues allow more than one solution; "
                "showing one. Fill a few cells and solve again to narrow it."
            )
        self.window.show_result(
            f"Towers puzzle solved (confidence {sol.read.confidence:.0%}).", body, pixmap
        )

    def _show_compass(self, read: "compass_mod.CompassRead"):
        pixmap = window_mod.render_compass(read.bearing_deg)
        body = (
            f"Walk <b>{read.wind}</b> (bearing {read.bearing_deg:.0f}\N{DEGREE SIGN}). "
            "Move a good distance, then solve again to re-read the needle."
            "<br>Automatic triangulation isn't supported yet."
        )
        self.window.show_result("Compass clue.", body, pixmap)

    def _show_text_result(self, result: "solver.SolveResult"):
        if result.status == "no_clue":
            if self._auto_solving:
                # screening false positive; don't clobber whatever is shown
                self.window.set_busy(False, "Watching for clues…")
                return
            self.window.show_message(
                "No clue interface found on screen. Open the clue scroll or puzzle and solve again.",
                _ocr_details(result),
            )
            return
        if result.status == "unsupported":
            text = f"Found “{result.title}” but couldn't match the clue."
            if result.read_text:
                text += f" Read: “{result.read_text}”"
            self.window.show_message(text, _ocr_details(result))
            return

        ratio, entry = result.best
        prefix = "" if result.status == "solved" else "Low confidence match. "
        answer = html.escape(solver.describe(entry)).replace("\n", "<br>")
        body = f"<b>“{html.escape(solver.clue_text(entry) or result.read_text)}”</b><br><br>{answer}"
        pixmap = None
        if result.map_image is not None:
            pixmap = window_mod.np_to_pixmap(result.map_image)
        self.window.show_result(
            f"{prefix}Clue matched ({ratio:.0%}).", body, pixmap, _ocr_details(result)
        )

    @Slot(object)
    def on_result(self, result):
        if isinstance(result, slide_mod.SlideSolution):
            self._show_slide(result)
        elif isinstance(result, lockbox_mod.LockboxSolution):
            self._show_lockbox(result)
        elif isinstance(result, towers_mod.TowersSolution):
            self._show_towers(result)
        elif isinstance(result, compass_mod.CompassRead):
            self._show_compass(result)
        else:
            self._show_text_result(result)

    @Slot(str)
    def on_failed(self, message: str):
        self.window.show_message(message)
