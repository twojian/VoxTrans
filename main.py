import sys
import os
import logging

from dotenv import load_dotenv

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from config import config_manager


def setup_logging():
    level = getattr(logging, config_manager.config.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    setup_logging()
    logger = logging.getLogger("VoxTrans")
    logger.info("Starting VoxTrans...")

    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt

    app = QApplication(sys.argv)
    app.setApplicationName("VoxTrans")
    app.setApplicationVersion("1.0.0")

    from ui.main_window import MainWindow

    window = MainWindow()
    window.show()

    logger.info("VoxTrans ready")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
