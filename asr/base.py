from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class ASRResult:
    text: str
    language: str = ""
    confidence: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    segments: List[dict] = field(default_factory=list)


class BaseASR(ABC):
    @abstractmethod
    def recognize(self, audio: np.ndarray, language: str = "auto") -> ASRResult:
        ...

    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        ...

    def close(self):
        pass


def create_asr(config) -> BaseASR:
    engine = config.engine
    if engine == "faster-whisper":
        from asr.faster_whisper_engine import FasterWhisperASR
        return FasterWhisperASR(config)
    elif engine == "whisper-api":
        from asr.whisper_api import WhisperAPIASR
        return WhisperAPIASR(config)
    else:
        raise ValueError(f"Unknown ASR engine: {engine}")
