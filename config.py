import json
import os
import importlib.util
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any


def _is_torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


TORCH_AVAILABLE = _is_torch_available()


def get_available_device() -> str:
    if not TORCH_AVAILABLE:
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk: int = 1024
    device_index: int = -1
    capture_mode: str = "system"  # "microphone" | "system"
    volume_gain: float = 1.5
    noise_reduction_strength: float = 0.3


@dataclass
class VADConfig:
    enabled: bool = True
    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 500
    frame_size: int = 512


@dataclass
class ASRConfig:
    engine: str = "faster-whisper"  # "faster-whisper" | "whisper-api"
    model_size: str = "base"  # tiny / base / small / medium / large
    language: str = "auto"
    device: str = "auto"  # auto / cuda / cpu
    compute_type: str = "int8"
    beam_size: int = 3
    # Whisper API
    api_key: str = ""
    api_base_url: str = "https://api.openai.com/v1"


@dataclass
class TranslationConfig:
    engine: str = "llm-api"  # "llm-api" | "local"
    target_language: str = "zh"
    # LLM API
    api_key: str = ""
    api_base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    # Local model
    local_model_name: str = "Helsinki-NLP/opus-mt-en-zh"
    # Context
    context_sentences: int = 5
    max_concurrent: int = 3


@dataclass
class SubtitleConfig:
    show_original: bool = True
    font_family: str = "Microsoft YaHei"
    font_size_original: int = 20
    font_size_translated: int = 28
    color_original: str = "#CCCCCC"
    color_translated: str = "#FFFFFF"
    bg_color: str = "#000000"
    bg_opacity: float = 0.6
    outline_width: int = 2
    outline_color: str = "#000000"
    max_lines: int = 3
    display_duration: float = 8.0
    window_width: int = 900
    window_height: int = 150


@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    subtitle: SubtitleConfig = field(default_factory=SubtitleConfig)
    output_dir: str = "./output"
    log_level: str = "INFO"


class ConfigManager:
    _instance: Optional["ConfigManager"] = None
    _config_file: str = "./config.json"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = AppConfig()
            cls._instance._load_config()
        return cls._instance

    @property
    def config(self) -> AppConfig:
        return self._config

    @property
    def audio(self) -> AudioConfig:
        return self._config.audio

    @property
    def vad(self) -> VADConfig:
        return self._config.vad

    @property
    def asr(self) -> ASRConfig:
        return self._config.asr

    @property
    def translation(self) -> TranslationConfig:
        return self._config.translation

    @property
    def subtitle(self) -> SubtitleConfig:
        return self._config.subtitle

    def _load_config(self):
        if not os.path.exists(self._config_file):
            return
        try:
            with open(self._config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "audio" in data:
                self._config.audio = AudioConfig(**data["audio"])
            if "vad" in data:
                self._config.vad = VADConfig(**data["vad"])
            if "asr" in data:
                self._config.asr = ASRConfig(**data["asr"])
            if "translation" in data:
                self._config.translation = TranslationConfig(**data["translation"])
            if "subtitle" in data:
                self._config.subtitle = SubtitleConfig(**data["subtitle"])
            for key in ("output_dir", "log_level"):
                if key in data:
                    setattr(self._config, key, data[key])
        except Exception as e:
            print(f"Failed to load config: {e}, using defaults")

        self._resolve_device()

    def _resolve_device(self):
        if self._config.asr.device == "auto":
            self._config.asr.device = get_available_device()

    def get_actual_device(self) -> str:
        if self._config.asr.device == "auto":
            return get_available_device()
        return self._config.asr.device

    def save_config(self):
        try:
            data = {
                "audio": asdict(self._config.audio),
                "vad": asdict(self._config.vad),
                "asr": asdict(self._config.asr),
                "translation": asdict(self._config.translation),
                "subtitle": asdict(self._config.subtitle),
                "output_dir": self._config.output_dir,
                "log_level": self._config.log_level,
            }
            with open(self._config_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save config: {e}")

    def reset_to_default(self):
        self._config = AppConfig()
        self._resolve_device()
        self.save_config()


config_manager = ConfigManager()
