from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QLabel, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox,
    QCheckBox, QPushButton, QGroupBox, QFormLayout,
    QDialogButtonBox,
)
from PyQt6.QtCore import Qt

from config import config_manager


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumSize(500, 400)

        layout = QVBoxLayout(self)
        tabs = QTabWidget()

        tabs.addTab(self._build_asr_tab(), "ASR")
        tabs.addTab(self._build_translation_tab(), "Translation")
        tabs.addTab(self._build_subtitle_tab(), "Subtitle")

        layout.addWidget(tabs)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _build_asr_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self.asr_engine = QComboBox()
        self.asr_engine.addItems(["faster-whisper", "whisper-api"])
        self.asr_engine.setCurrentText(config_manager.asr.engine)
        form.addRow("Engine:", self.asr_engine)

        self.asr_model = QComboBox()
        self.asr_model.addItems(["tiny", "base", "small", "medium", "large"])
        self.asr_model.setCurrentText(config_manager.asr.model_size)
        form.addRow("Model Size:", self.asr_model)

        self.asr_device = QComboBox()
        self.asr_device.addItems(["auto", "cuda", "cpu"])
        self.asr_device.setCurrentText(config_manager.asr.device)
        form.addRow("Device:", self.asr_device)

        self.asr_api_key = QLineEdit(config_manager.asr.api_key)
        self.asr_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("API Key:", self.asr_api_key)

        self.asr_api_url = QLineEdit(config_manager.asr.api_base_url)
        form.addRow("API Base URL:", self.asr_api_url)

        return w

    def _build_translation_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self.trans_engine = QComboBox()
        self.trans_engine.addItems(["llm-api", "local"])
        self.trans_engine.setCurrentText(config_manager.translation.engine)
        form.addRow("Engine:", self.trans_engine)

        self.trans_model = QLineEdit(config_manager.translation.model)
        form.addRow("Model:", self.trans_model)

        self.trans_api_key = QLineEdit(config_manager.translation.api_key)
        self.trans_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("API Key:", self.trans_api_key)

        self.trans_api_url = QLineEdit(config_manager.translation.api_base_url)
        form.addRow("API Base URL:", self.trans_api_url)

        self.trans_local_model = QLineEdit(config_manager.translation.local_model_name)
        form.addRow("Local Model:", self.trans_local_model)

        self.trans_context = QSpinBox()
        self.trans_context.setRange(0, 20)
        self.trans_context.setValue(config_manager.translation.context_sentences)
        form.addRow("Context Sentences:", self.trans_context)

        return w

    def _build_subtitle_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self.sub_show_orig = QCheckBox("Show original text")
        self.sub_show_orig.setChecked(config_manager.subtitle.show_original)
        form.addRow(self.sub_show_orig)

        self.sub_font = QLineEdit(config_manager.subtitle.font_family)
        form.addRow("Font:", self.sub_font)

        self.sub_size_orig = QSpinBox()
        self.sub_size_orig.setRange(8, 72)
        self.sub_size_orig.setValue(config_manager.subtitle.font_size_original)
        form.addRow("Original Font Size:", self.sub_size_orig)

        self.sub_size_trans = QSpinBox()
        self.sub_size_trans.setRange(8, 72)
        self.sub_size_trans.setValue(config_manager.subtitle.font_size_translated)
        form.addRow("Translation Font Size:", self.sub_size_trans)

        self.sub_opacity = QDoubleSpinBox()
        self.sub_opacity.setRange(0.0, 1.0)
        self.sub_opacity.setSingleStep(0.1)
        self.sub_opacity.setValue(config_manager.subtitle.bg_opacity)
        form.addRow("Background Opacity:", self.sub_opacity)

        self.sub_max_lines = QSpinBox()
        self.sub_max_lines.setRange(1, 10)
        self.sub_max_lines.setValue(config_manager.subtitle.max_lines)
        form.addRow("Max Lines:", self.sub_max_lines)

        return w

    def _save_and_accept(self):
        # ASR
        config_manager.asr.engine = self.asr_engine.currentText()
        config_manager.asr.model_size = self.asr_model.currentText()
        config_manager.asr.device = self.asr_device.currentText()
        config_manager.asr.api_key = self.asr_api_key.text()
        config_manager.asr.api_base_url = self.asr_api_url.text()

        # Translation
        config_manager.translation.engine = self.trans_engine.currentText()
        config_manager.translation.model = self.trans_model.text()
        config_manager.translation.api_key = self.trans_api_key.text()
        config_manager.translation.api_base_url = self.trans_api_url.text()
        config_manager.translation.local_model_name = self.trans_local_model.text()
        config_manager.translation.context_sentences = self.trans_context.value()

        # Subtitle
        config_manager.subtitle.show_original = self.sub_show_orig.isChecked()
        config_manager.subtitle.font_family = self.sub_font.text()
        config_manager.subtitle.font_size_original = self.sub_size_orig.value()
        config_manager.subtitle.font_size_translated = self.sub_size_trans.value()
        config_manager.subtitle.bg_opacity = self.sub_opacity.value()
        config_manager.subtitle.max_lines = self.sub_max_lines.value()

        config_manager.save_config()
        self.accept()
