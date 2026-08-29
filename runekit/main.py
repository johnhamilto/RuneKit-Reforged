import logging
import logging.handlers
import os
import signal
import sys
import traceback

import click
from PySide6.QtCore import QSettings, QStandardPaths, Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QMessageBox,
)

import runekit._resources
from runekit import browser
from runekit.game import get_platform_manager
from runekit.host import Host


def setup_logging():
    """Configure logging to both console and a rotating log file."""
    log_dir = os.path.join(
        QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation),
        "logs",
    )
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "runekit.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # PIL logs every TIFF tag of every capture at DEBUG; that floods the file
    logging.getLogger("PIL").setLevel(logging.INFO)

    # Console: INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    root_logger.addHandler(console_handler)

    # File: DEBUG and above, rotating 3 x 1MB
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root_logger.addHandler(file_handler)

    logging.info("Log file: %s", log_file)


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
    )
)
@click.option("--game-index", default=0, help="Game instance index, starting from 0")
@click.argument("qt_args", nargs=-1, type=click.UNPROCESSED)
@click.argument("app_url", required=False)
def main(app_url, game_index, qt_args):
    setup_logging()

    logging.info("Starting QtWebEngine")
    browser.init()

    app = QApplication(["runekit", *qt_args])
    app.setQuitOnLastWindowClosed(False)
    app.setOrganizationName("cupco.de")
    app.setOrganizationDomain("cupco.de")
    app.setApplicationName("RuneKit")

    signal.signal(signal.SIGINT, lambda no, frame: app.quit())

    timer = QTimer()
    timer.start(300)
    timer.timeout.connect(lambda: None)

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)

    game_manager = None
    try:
        game_manager = get_platform_manager()
        host = Host(game_manager)

        if app_url == "settings":
            host.open_settings()
            host.setting_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            host.setting_dialog.destroyed.connect(app.quit)
        elif app_url:
            logging.info("Loading app")
            game_app = host.launch_app_from_url(app_url)
            game_app.window.destroyed.connect(app.quit)
        else:
            if not host.app_store.has_default_apps():
                host.app_store.load_default_apps()

        app.exec()
        sys.exit(0)
    except Exception as e:
        msg = QMessageBox(
            QMessageBox.Icon.Critical,
            "Oh No!",
            f"Fatal error: \n\n{e.__class__.__name__}: {e}",
        )
        msg.setDetailedText(traceback.format_exc())
        msg.exec()

        raise
    finally:
        if game_manager is not None:
            logging.debug("Stopping game manager")
            game_manager.stop()


if __name__ == "__main__":
    main()
