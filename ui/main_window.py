import logging
from datetime import datetime

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QTextEdit, QStatusBar, QProgressBar,
    QMenuBar, QMenu, QGroupBox, QSplitter, QFileDialog, QMessageBox,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QFont, QAction, QTextCursor, QColor

from config import config_manager
from audio.capture import AudioCapture
from pipeline.interpreter import InterpreterPipeline

logger = logging.getLogger(__name__)

LANGUAGES = [
    ("auto", "Auto Detect"),
    ("en", "English"),
    ("zh", "Chinese"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("fr", "French"),
    ("de", "German"),
    ("es", "Spanish"),
    ("ru", "Russian"),
]

TARGET_LANGUAGES = [
    ("zh", "Chinese"),
    ("en", "English"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VoxTrans - AI Simultaneous Interpreter")
        self.setMinimumSize(700, 500)
        self.resize(900, 600)

        self.pipeline = InterpreterPipeline()
        self.floating_window = None
        self._history = []  # [(original, translated, timestamp)]

        self._setup_menu()
        self._setup_ui()
        self._setup_statusbar()
        self._connect_signals()

        self._refresh_devices()

    def _setup_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        export_act = QAction("Export Subtitles...", self)
        export_act.triggered.connect(self._export_subtitles)
        file_menu.addAction(export_act)
        file_menu.addSeparator()
        quit_act = QAction("Quit", self)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        view_menu = menubar.addMenu("View")
        float_act = QAction("Show Floating Subtitle", self)
        float_act.triggered.connect(self._toggle_floating)
        view_menu.addAction(float_act)

        settings_menu = menubar.addMenu("Settings")
        settings_act = QAction("Preferences...", self)
        settings_act.triggered.connect(self._open_settings)
        settings_menu.addAction(settings_act)

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # --- Control bar ---
        control_group = QGroupBox("Controls")
        control_layout = QHBoxLayout(control_group)

        control_layout.addWidget(QLabel("Audio:"))
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(200)
        control_layout.addWidget(self.device_combo)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_devices)
        control_layout.addWidget(self.refresh_btn)

        control_layout.addWidget(QLabel("From:"))
        self.source_lang_combo = QComboBox()
        for code, name in LANGUAGES:
            self.source_lang_combo.addItem(name, code)
        control_layout.addWidget(self.source_lang_combo)

        control_layout.addWidget(QLabel("To:"))
        self.target_lang_combo = QComboBox()
        for code, name in TARGET_LANGUAGES:
            self.target_lang_combo.addItem(name, code)
        control_layout.addWidget(self.target_lang_combo)

        control_layout.addStretch()

        self.start_btn = QPushButton("Start")
        self.start_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "padding: 6px 20px; font-weight: bold; border-radius: 4px; }"
            "QPushButton:hover { background-color: #45a049; }"
        )
        self.start_btn.clicked.connect(self._toggle_start)
        control_layout.addWidget(self.start_btn)

        main_layout.addWidget(control_group)

        # --- Audio level ---
        level_layout = QHBoxLayout()
        level_layout.addWidget(QLabel("Audio Level:"))
        self.level_bar = QProgressBar()
        self.level_bar.setMaximum(100)
        self.level_bar.setTextVisible(False)
        self.level_bar.setMaximumHeight(12)
        level_layout.addWidget(self.level_bar)
        main_layout.addLayout(level_layout)

        # --- Subtitle display ---
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Original text
        orig_group = QGroupBox("Original")
        orig_layout = QVBoxLayout(orig_group)
        self.original_text = QTextEdit()
        self.original_text.setReadOnly(True)
        self.original_text.setFont(QFont("Consolas", 12))
        orig_layout.addWidget(self.original_text)
        splitter.addWidget(orig_group)

        # Translated text
        trans_group = QGroupBox("Translation")
        trans_layout = QVBoxLayout(trans_group)
        self.translated_text = QTextEdit()
        self.translated_text.setReadOnly(True)
        self.translated_text.setFont(QFont("Microsoft YaHei", 14))
        trans_layout.addWidget(self.translated_text)
        splitter.addWidget(trans_group)

        splitter.setSizes([400, 400])
        main_layout.addWidget(splitter, stretch=1)

    def _setup_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.status_label = QLabel("Ready")
        self.statusbar.addWidget(self.status_label)

    def _connect_signals(self):
        self.pipeline.subtitle_updated.connect(self._on_subtitle)
        self.pipeline.status_changed.connect(self._on_status)
        self.pipeline.error_occurred.connect(self._on_error)
        self.pipeline.audio_level.connect(self._on_audio_level)

    def _refresh_devices(self):
        self.device_combo.clear()
        cap = AudioCapture()
        devices = cap.list_devices()
        cap.close()
        for dev in devices:
            label = f"[{dev['index']}] {dev['name']}"
            if dev["is_system_audio"]:
                label += " (System)"
            elif dev["is_microphone"]:
                label += " (Mic)"
            self.device_combo.addItem(label, dev["index"])

    def _toggle_start(self):
        if self.pipeline.is_running:
            self.pipeline.stop()
            self.start_btn.setText("Start")
            self.start_btn.setStyleSheet(
                "QPushButton { background-color: #4CAF50; color: white; "
                "padding: 6px 20px; font-weight: bold; border-radius: 4px; }"
            )
        else:
            idx = self.device_combo.currentData()
            if idx is None:
                QMessageBox.warning(self, "Error", "Please select an audio device.")
                return
            self.pipeline.set_device(idx)
            source = self.source_lang_combo.currentData()
            target = self.target_lang_combo.currentData()
            self.pipeline.start(source_lang=source, target_lang=target)
            self.start_btn.setText("Stop")
            self.start_btn.setStyleSheet(
                "QPushButton { background-color: #f44336; color: white; "
                "padding: 6px 20px; font-weight: bold; border-radius: 4px; }"
            )

    @pyqtSlot(str, str)
    def _on_subtitle(self, original: str, translated: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self._history.append((original, translated, ts))

        self.original_text.append(f"[{ts}] {original}")
        self.original_text.moveCursor(QTextCursor.MoveOperation.End)

        self.translated_text.append(f"[{ts}] {translated}")
        self.translated_text.moveCursor(QTextCursor.MoveOperation.End)

        # Update floating window
        if self.floating_window and self.floating_window.isVisible():
            self.floating_window.update_subtitle(original, translated)

    @pyqtSlot(str)
    def _on_status(self, status: str):
        self.status_label.setText(status)

    @pyqtSlot(str)
    def _on_error(self, error: str):
        QMessageBox.critical(self, "Error", error)
        self.start_btn.setText("Start")

    @pyqtSlot(float)
    def _on_audio_level(self, rms: float):
        level = min(int(rms * 500), 100)
        self.level_bar.setValue(level)

    def _toggle_floating(self):
        try:
            from ui.floating_subtitle import FloatingSubtitleWindow
            if self.floating_window is None:
                self.floating_window = FloatingSubtitleWindow()
            if self.floating_window.isVisible():
                self.floating_window.hide()
            else:
                self.floating_window.show()
        except ImportError:
            QMessageBox.information(self, "Info", "Floating subtitle module not available yet.")

    def _open_settings(self):
        try:
            from ui.settings_dialog import SettingsDialog
            dlg = SettingsDialog(self)
            if dlg.exec():
                config_manager.save_config()
        except ImportError:
            QMessageBox.information(self, "Info", "Settings module not available yet.")

    def _export_subtitles(self):
        if not self._history:
            QMessageBox.information(self, "Info", "No subtitles to export.")
            return
        try:
            from subtitle.exporter import SubtitleExporter
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Subtitles", "subtitles.srt",
                "SRT Files (*.srt);;VTT Files (*.vtt);;Text Files (*.txt)"
            )
            if path:
                SubtitleExporter.export(self._history, path)
                QMessageBox.information(self, "Success", f"Exported to {path}")
        except ImportError:
            QMessageBox.information(self, "Info", "Export module not available yet.")

    def closeEvent(self, event):
        self.pipeline.close()
        if self.floating_window:
            self.floating_window.close()
        event.accept()
