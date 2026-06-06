import numpy as np
import logging
from typing import List

from asr.base import BaseASR, ASRResult

logger = logging.getLogger(__name__)


class FasterWhisperASR(BaseASR):
    def __init__(self, config):
        self.config = config
        self.model = None
        self._load_model()

    def _load_model(self):
        from faster_whisper import WhisperModel

        device = self.config.device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        logger.info(
            f"Loading faster-whisper model: {self.config.model_size} "
            f"(device={device}, compute_type={self.config.compute_type})"
        )

        self.model = WhisperModel(
            self.config.model_size,
            device=device,
            compute_type=self.config.compute_type,
        )
        logger.info("Model loaded successfully")

    def recognize(self, audio: np.ndarray, language: str = "auto") -> ASRResult:
        if self.model is None:
            return ASRResult(text="", language="")

        if len(audio) == 0:
            return ASRResult(text="", language="")

        lang = None if language == "auto" else language

        segments_gen, info = self.model.transcribe(
            audio,
            language=lang,
            beam_size=self.config.beam_size,
            vad_filter=True,
        )

        segments = []
        texts = []
        for seg in segments_gen:
            texts.append(seg.text.strip())
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
            })

        text = " ".join(texts)
        detected_lang = info.language if info else (language if language != "auto" else "")

        return ASRResult(
            text=text,
            language=detected_lang,
            confidence=info.language_probability if info else 0.0,
            start_time=segments[0]["start"] if segments else 0.0,
            end_time=segments[-1]["end"] if segments else 0.0,
            segments=segments,
        )

    def get_supported_languages(self) -> List[str]:
        return [
            "en", "zh", "ja", "ko", "fr", "de", "es", "pt", "ru",
            "it", "nl", "pl", "tr", "ar", "th", "vi", "id", "hi",
        ]

    def close(self):
        self.model = None
