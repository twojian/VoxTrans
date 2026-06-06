import numpy as np
import io
import logging
from typing import List

from asr.base import BaseASR, ASRResult

logger = logging.getLogger(__name__)


class WhisperAPIASR(BaseASR):
    """Cloud-based ASR engine using OpenAI Whisper API."""

    def __init__(self, config):
        self.config = config
        import openai
        self.client = openai.OpenAI(
            api_key=config.api_key or self._get_env_key(),
            base_url=config.api_base_url or "https://api.openai.com/v1",
        )

    @staticmethod
    def _get_env_key() -> str:
        import os
        return os.environ.get("ASR_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

    def recognize(self, audio: np.ndarray, language: str = "auto") -> ASRResult:
        if len(audio) == 0:
            return ASRResult(text="", language="")

        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, audio, 16000, format="WAV")
        buf.seek(0)
        buf.name = "audio.wav"

        kwargs = {"model": "whisper-1", "file": buf, "response_format": "verbose_json"}
        if language != "auto":
            kwargs["language"] = language

        try:
            response = self.client.audio.transcriptions.create(**kwargs)

            text = response.text if hasattr(response, "text") else str(response)
            detected_lang = getattr(response, "language", language if language != "auto" else "")
            segments = []
            if hasattr(response, "segments") and response.segments:
                for seg in response.segments:
                    segments.append({
                        "start": seg.get("start", 0),
                        "end": seg.get("end", 0),
                        "text": seg.get("text", ""),
                    })

            return ASRResult(
                text=text,
                language=detected_lang,
                confidence=1.0,
                start_time=segments[0]["start"] if segments else 0.0,
                end_time=segments[-1]["end"] if segments else 0.0,
                segments=segments,
            )
        except Exception as e:
            logger.error(f"Whisper API error: {e}")
            return ASRResult(text="", language="")

    def get_supported_languages(self) -> List[str]:
        return [
            "en", "zh", "ja", "ko", "fr", "de", "es", "pt", "ru",
            "it", "nl", "pl", "tr", "ar", "th", "vi", "id", "hi",
        ]
