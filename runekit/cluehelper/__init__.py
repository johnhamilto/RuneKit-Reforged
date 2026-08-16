"""Built-in clue solver for treasure trail clues."""
import html
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QStandardPaths, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QFont, QPen
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsSimpleTextItem

from runekit import detection
from runekit.cluehelper import compass as compass_mod
from runekit.cluehelper import lockbox as lockbox_mod
from runekit.cluehelper import maps as maps_mod
from runekit.cluehelper import slide as slide_mod
from runekit.cluehelper import slide_solver, solver
from runekit.cluehelper import towers as towers_mod
from runekit.cluehelper import window as window_mod
from runekit.cluehelper.assets import ClueAssets

if TYPE_CHECKING:
    from runekit.game import GameInstance

logger = logging.getLogger(__name__)

OVERLAY_MOVES = 25
OVERLAY_TIMEOUT_MS = 20000


class _SolveThread(QThread):
    ok = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, instance: "GameInstance", cache_dir: Path, parent=None):
        super().__init__(parent=parent)
        self.instance = instance
        self.cache_dir = cache_dir

    def _hint_path(self) -> Path:
        return self.cache_dir / "scale_hint.json"

    def _load_hint(self):
        try:
            return json.loads(self._hint_path().read_text())["scale"]
        except Exception:
            return None

    def _save_hint(self, scale: float):
        try:
            self._hint_path().write_text(json.dumps({"scale": scale}))
        except Exception:
            pass

    def _puzzle_title(self, result) -> str:
        for line in result.lines:
            for title in ("lockbox", "towers", "celtic knot"):
                if solver._ratio(line["text"], title) >= 0.75:
                    return title
        return ""

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
                    result.status = "solved"
                    result.read_text = "(map image)"
                    result.matches = [(1.0, match.entry)]
                self.ok.emit(result)
                return
            if result.status != "no_clue":
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
            board = slide_mod.read_slide(arr, assets, hint=hint)
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

    @property
    def window(self) -> "window_mod.ClueSolverWindow":
        if self._window is None:
            self._window = window_mod.ClueSolverWindow()
            self._window.solve_requested.connect(self.solve_requested)
        return self._window

    def show_error(self, message: str):
        self.window.open()
        self.window.show_message(message)

    @Slot()
    def solve(self, instance: "GameInstance"):
        self.window.open()
        if self._thread is not None and self._thread.isRunning():
            return

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
            self.window.show_message(
                "No clue interface found on screen. Open the clue scroll or puzzle and solve again.",
                _ocr_details(result),
            )
            return
        if result.status == "unsupported":
            text = f"Found “{html.escape(result.title)}” but couldn't match the clue."
            if result.read_text:
                text += f" Read: “{html.escape(result.read_text)}”"
            self.window.show_message(text, _ocr_details(result))
            return

        ratio, entry = result.best
        prefix = "" if result.status == "solved" else "Low confidence match. "
        answer = html.escape(solver.describe(entry)).replace("\n", "<br>")
        body = f"<b>“{html.escape(solver.clue_text(entry) or result.read_text)}”</b><br><br>{answer}"
        self.window.show_result(
            f"{prefix}Clue matched ({ratio:.0%}).", body, None, _ocr_details(result)
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
