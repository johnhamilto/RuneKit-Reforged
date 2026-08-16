"""Built-in clue solver for treasure trail clues."""
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QStandardPaths, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QFont, QPen
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsSimpleTextItem,
    QMessageBox,
)

from runekit import detection
from runekit.cluehelper import lockbox as lockbox_mod
from runekit.cluehelper import slide as slide_mod
from runekit.cluehelper import slide_solver, solver
from runekit.cluehelper import towers as towers_mod
from runekit.cluehelper.assets import ClueAssets

if TYPE_CHECKING:
    from runekit.game import GameInstance

logger = logging.getLogger(__name__)

OVERLAY_MOVES = 25
OVERLAY_TIMEOUT_MS = 20000


class _SolveThread(QThread):
    ok = Signal(object)
    failed = Signal(str)

    def __init__(self, instance: "GameInstance", cache_dir: Path, parent=None):
        super().__init__(parent=parent)
        self.instance = instance
        self.cache_dir = cache_dir

    def _puzzle_title(self, result) -> str:
        for line in result.lines:
            for title in ("lockbox", "towers", "celtic knot"):
                if solver._ratio(line["text"], title) >= 0.75:
                    return title
        return ""

    def run(self):
        try:
            frame = self.instance.grab_game()
            dbs = solver.load_databases(self.cache_dir)
            result = solver.solve_frame(frame, dbs)
            if result.status != "no_clue":
                self.ok.emit(result)
                return

            arr = detection.to_array(frame)
            assets = ClueAssets(self.cache_dir)
            title = self._puzzle_title(result)

            if title == "lockbox":
                read = lockbox_mod.read_lockbox(arr, assets)
                if read is None:
                    self.failed.emit(
                        "Lockbox detected but the grid could not be read.\n"
                        "Make sure the whole box is visible and try again."
                    )
                    return
                sol = lockbox_mod.solve_lockbox(read.grid)
                if sol is None:
                    self.failed.emit(
                        "Lockbox grid read but no solution exists; likely a misread. Try again."
                    )
                    return
                self.ok.emit(lockbox_mod.LockboxSolution(read, sol[0], sol[1]))
                return

            if title == "towers":
                read = towers_mod.read_towers(arr, assets)
                if read is None:
                    self.failed.emit(
                        "Towers puzzle detected but the clues could not be read.\n"
                        "Make sure the whole board is visible and try again."
                    )
                    return
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

            board = slide_mod.read_slide(arr, assets)
            if board is not None:
                try:
                    moves = slide_solver.solve(board.board)
                except ValueError as e:
                    self.failed.emit(
                        f"Slide puzzle detected but the board read is inconsistent ({e}).\n"
                        "Make sure no tile is mid-animation and try again."
                    )
                    return
                self.ok.emit(slide_mod.SlideSolution(board, moves))
                return

            self.ok.emit(result)
        except Exception as e:
            logger.error("Clue solve failed", exc_info=True)
            self.failed.emit(str(e))


class ClueHelper(QObject):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.cache_dir = Path(
            QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation)
        )
        self._thread = None
        self._instance = None

    @Slot()
    def solve(self, instance: "GameInstance"):
        if self._thread is not None and self._thread.isRunning():
            return

        self._instance = instance
        self._thread = _SolveThread(instance, self.cache_dir, parent=self)
        self._thread.ok.connect(self.on_result)
        self._thread.failed.connect(self.on_failed)
        self._thread.start()

    def _draw_slide_moves(self, sol: "slide_mod.SlideSolution") -> bool:
        try:
            area = self._instance.get_overlay_area()
        except Exception:
            return False
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

        def cleanup():
            for item in items:
                scene = item.scene()
                if scene is not None:
                    scene.removeItem(item)

        QTimer.singleShot(OVERLAY_TIMEOUT_MS, cleanup)
        return True

    def _show_slide(self, sol: "slide_mod.SlideSolution"):
        drew = self._draw_slide_moves(sol)
        msg = QMessageBox(QMessageBox.Icon.Information, "Clue Solver", "")
        text = (
            f"Slide puzzle read (theme {sol.board.theme}, "
            f"confidence {sol.board.confidence:.0%}).\n\n"
            f"Solution: {len(sol.moves)} moves."
        )
        if drew:
            text += f"\n\nThe first {min(OVERLAY_MOVES, len(sol.moves))} clicks are numbered on the game. Solve again for the next batch."
        msg.setText(text)
        msg.setDetailedText(
            "Full click sequence (row, column), 1-based:\n"
            + " ".join(f"({cell // 5 + 1},{cell % 5 + 1})" for cell in sol.moves)
        )
        msg.exec()

    def _draw_grid_numbers(self, origin, stride, values, color=QColor(255, 220, 80)) -> bool:
        """Draw per-cell numbers over a 5x5 grid on the game overlay."""
        try:
            area = self._instance.get_overlay_area()
        except Exception:
            return False
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

        def cleanup():
            for item in items:
                scene = item.scene()
                if scene is not None:
                    scene.removeItem(item)

        QTimer.singleShot(OVERLAY_TIMEOUT_MS, cleanup)
        return True

    def _show_lockbox(self, sol: "lockbox_mod.LockboxSolution"):
        stride = lockbox_mod.TILE * sol.read.scale
        drew = self._draw_grid_numbers(sol.read.origin, stride, sol.presses)
        total = sum(map(sum, sol.presses))
        msg = QMessageBox(QMessageBox.Icon.Information, "Clue Solver", "")
        text = (
            f"Lockbox read (confidence {sol.read.confidence:.0%}).\n\n"
            f"{total} presses to make every tile "
            f"{lockbox_mod.TILE_NAMES[sol.target]}."
        )
        if drew:
            text += "\n\nPress each tile the number of times shown on the game."
        msg.setText(text)
        msg.setDetailedText(
            "Press counts per tile (row by row):\n"
            + "\n".join(" ".join(str(v) for v in row) for row in sol.presses)
        )
        msg.exec()

    def _show_towers(self, sol: "towers_mod.TowersSolution"):
        s = sol.read.scale
        ix, iy = sol.read.inner_origin
        origin = (ix + 27 * s, iy + 27.5 * s)
        drew = self._draw_grid_numbers(origin, 44 * s, sol.grid, color=QColor(120, 235, 120))
        msg = QMessageBox(QMessageBox.Icon.Information, "Clue Solver", "")
        text = f"Towers puzzle solved (confidence {sol.read.confidence:.0%})."
        if sol.solutions > 1:
            text += "\n\nThe visible clues allow more than one solution; showing one. Fill a few cells and solve again to narrow it."
        if drew:
            text += "\n\nThe solution is drawn on the game."
        msg.setText(text)
        msg.setDetailedText(
            "Solution:\n" + "\n".join(" ".join(str(v) for v in row) for row in sol.grid)
        )
        msg.exec()

    @Slot(object)
    def on_result(self, result):
        if isinstance(result, slide_mod.SlideSolution):
            self._show_slide(result)
            return
        if isinstance(result, lockbox_mod.LockboxSolution):
            self._show_lockbox(result)
            return
        if isinstance(result, towers_mod.TowersSolution):
            self._show_towers(result)
            return
        msg = QMessageBox(QMessageBox.Icon.Information, "Clue Solver", "")
        msg.setDetailedText(
            "Text found on screen:\n"
            + "\n".join(f"{l['text']}  ({l['confidence']:.2f})" for l in result.lines)
        )

        if result.status == "no_clue":
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setText("No clue interface found on screen.\n\nOpen the clue scroll and try again.")
        elif result.status == "unsupported":
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setText(
                f"Found “{result.title}” but couldn't match the clue text.\n\n"
                "Map, coordinate, scan and puzzle clues aren't supported yet."
                + (f"\n\nRead: {result.read_text}" if result.read_text else "")
            )
        else:
            ratio, entry = result.best
            prefix = "" if result.status == "solved" else "Low confidence match.\n\n"
            msg.setText(
                f"{prefix}“{solver.clue_text(entry)}”\n\n"
                f"{solver.describe(entry)}\n\nMatch: {ratio:.0%}"
            )

        msg.exec()

    @Slot(str)
    def on_failed(self, message: str):
        QMessageBox.critical(None, "Clue Solver", f"Could not solve clue:\n\n{message}")
