from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QMenu
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QFont, QColor, QPainter, QAction, QCursor

from config import config_manager


class FloatingSubtitleWindow(QWidget):
    """Transparent floating window for bilingual subtitle overlay."""

    def __init__(self):
        super().__init__()
        cfg = config_manager.subtitle

        self.setWindowTitle("VoxTrans Subtitle")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(cfg.window_width, cfg.window_height)

        # Drag state
        self._dragging = False
        self._drag_offset = QPoint()
        self._locked = False

        self._setup_ui(cfg)

        # Position at bottom center of screen
        from PyQt6.QtWidgets import QApplication
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            x = (geo.width() - self.width()) // 2
            y = geo.height() - self.height() - 60
            self.move(x, y)

    def _setup_ui(self, cfg):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 10, 20, 10)

        # Original text (smaller, dimmer)
        self.original_label = QLabel("")
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.original_label.setWordWrap(True)
        self.original_label.setFont(QFont(cfg.font_family, cfg.font_size_original))
        self.original_label.setStyleSheet(f"color: {cfg.color_original};")
        layout.addWidget(self.original_label)

        # Translated text (larger, brighter)
        self.translated_label = QLabel("")
        self.translated_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.translated_label.setWordWrap(True)
        self.translated_label.setFont(
            QFont(cfg.font_family, cfg.font_size_translated, QFont.Weight.Bold)
        )
        self.translated_label.setStyleSheet(f"color: {cfg.color_translated};")
        layout.addWidget(self.translated_label)

    def update_subtitle(self, original: str, translated: str):
        if config_manager.subtitle.show_original:
            self.original_label.setText(original)
            self.original_label.setVisible(True)
        else:
            self.original_label.setVisible(False)
        self.translated_label.setText(translated)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        cfg = config_manager.subtitle
        bg = QColor(cfg.bg_color)
        bg.setAlphaF(cfg.bg_opacity)
        painter.setBrush(bg)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 10, 10)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and not self._locked:
            self._dragging = True
            self._drag_offset = event.globalPosition().toPoint() - self.pos()

    def mouseMoveEvent(self, event):
        if self._dragging:
            self.move(event.globalPosition().toPoint() - self._drag_offset)

    def mouseReleaseEvent(self, event):
        self._dragging = False

    def contextMenuEvent(self, event):
        menu = QMenu(self)

        show_orig = QAction(
            "Hide Original" if self.original_label.isVisible() else "Show Original",
            self,
        )
        show_orig.triggered.connect(self._toggle_original)
        menu.addAction(show_orig)

        lock_act = QAction(
            "Unlock Position" if self._locked else "Lock Position", self
        )
        lock_act.triggered.connect(self._toggle_lock)
        menu.addAction(lock_act)

        menu.addSeparator()
        close_act = QAction("Close", self)
        close_act.triggered.connect(self.hide)
        menu.addAction(close_act)

        menu.exec(QCursor.pos())

    def _toggle_original(self):
        config_manager.subtitle.show_original = not config_manager.subtitle.show_original
        self.original_label.setVisible(config_manager.subtitle.show_original)

    def _toggle_lock(self):
        self._locked = not self._locked
