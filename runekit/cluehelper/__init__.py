"""Built-in clue solver for text-based treasure trail clues."""
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QStandardPaths, QThread, Signal, Slot
from PySide6.QtWidgets import QMessageBox

from runekit.cluehelper import solver

if TYPE_CHECKING:
    from runekit.game import GameInstance

logger = logging.getLogger(__name__)


class _SolveThread(QThread):
    ok = Signal(object)
    failed = Signal(str)

    def __init__(self, instance: "GameInstance", cache_dir: Path, parent=None):
        super().__init__(parent=parent)
        self.instance = instance
        self.cache_dir = cache_dir

    def run(self):
        try:
            frame = self.instance.grab_game()
            dbs = solver.load_databases(self.cache_dir)
            self.ok.emit(solver.solve_frame(frame, dbs))
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

    @Slot()
    def solve(self, instance: "GameInstance"):
        if self._thread is not None and self._thread.isRunning():
            return

        self._thread = _SolveThread(instance, self.cache_dir, parent=self)
        self._thread.ok.connect(self.on_result)
        self._thread.failed.connect(self.on_failed)
        self._thread.start()

    @Slot(object)
    def on_result(self, result: "solver.SolveResult"):
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
