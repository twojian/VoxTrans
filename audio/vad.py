import numpy as np
import logging
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

_silero_available = None


def is_silero_available() -> bool:
    global _silero_available
    if _silero_available is not None:
        return _silero_available
    try:
        import torch
        from silero_vad import load_silero_vad
        _silero_available = True
    except ImportError:
        _silero_available = False
    return _silero_available


class SileroVAD:
    SUPPORTED_SAMPLE_RATES = [8000, 16000]
    FRAME_SIZE = 512

    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000,
                 min_speech_ms: int = 250, min_silence_ms: int = 500):
        if sample_rate not in self.SUPPORTED_SAMPLE_RATES:
            raise ValueError(f"Silero VAD only supports {self.SUPPORTED_SAMPLE_RATES}")

        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_frames = max(1, int(min_speech_ms / (self.FRAME_SIZE * 1000 / sample_rate)))
        self.min_silence_frames = max(1, int(min_silence_ms / (self.FRAME_SIZE * 1000 / sample_rate)))

        self._buffer = np.array([], dtype=np.float32)
        self.speech_state = False
        self.speech_count = 0
        self.silence_count = 0
        self.last_prob = 0.0

        import torch
        from silero_vad import load_silero_vad
        self._torch = torch
        self.model = load_silero_vad()

        if torch.cuda.is_available():
            try:
                self.model = self.model.to("cuda")
                self._device = "cuda"
            except Exception:
                self._device = "cpu"
        else:
            self._device = "cpu"

        logger.info(f"Silero VAD loaded (device={self._device}, threshold={threshold})")

    def _process_frame(self, frame: np.ndarray) -> Tuple[bool, float]:
        tensor = self._torch.from_numpy(frame)
        if self._device == "cuda":
            tensor = tensor.to("cuda")
        prob = self.model(tensor, self.sample_rate).item()
        return prob > self.threshold, prob

    def process_chunk(self, audio: np.ndarray) -> bool:
        self._buffer = np.concatenate([self._buffer, audio])

        while len(self._buffer) >= self.FRAME_SIZE:
            frame = self._buffer[:self.FRAME_SIZE]
            self._buffer = self._buffer[self.FRAME_SIZE:]

            is_speech, prob = self._process_frame(frame)
            self.last_prob = prob

            if is_speech:
                self.silence_count = 0
                self.speech_count += 1
                if self.speech_count >= self.min_speech_frames:
                    self.speech_state = True
            else:
                self.speech_count = 0
                if self.speech_state:
                    self.silence_count += 1
                    if self.silence_count >= self.min_silence_frames:
                        self.speech_state = False

        return self.speech_state

    def reset(self):
        self._buffer = np.array([], dtype=np.float32)
        self.speech_state = False
        self.speech_count = 0
        self.silence_count = 0
        self.last_prob = 0.0
        self.model.reset_states()


class EnergyVAD:
    def __init__(self, threshold: float = 0.015, sample_rate: int = 16000,
                 min_speech_ms: int = 250, min_silence_ms: int = 500):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.frame_ms = 30
        self.frame_size = int(sample_rate * self.frame_ms / 1000)
        self.min_speech_frames = max(1, min_speech_ms // self.frame_ms)
        self.min_silence_frames = max(1, min_silence_ms // self.frame_ms)
        self.speech_state = False
        self.speech_count = 0
        self.silence_count = 0

    def process_chunk(self, audio: np.ndarray) -> bool:
        n_frames = len(audio) // self.frame_size
        for i in range(n_frames):
            frame = audio[i * self.frame_size:(i + 1) * self.frame_size]
            energy = np.sum(frame ** 2) / len(frame)
            is_speech = energy > self.threshold

            if is_speech:
                self.silence_count = 0
                self.speech_count += 1
                if self.speech_count >= self.min_speech_frames:
                    self.speech_state = True
            else:
                self.speech_count = 0
                if self.speech_state:
                    self.silence_count += 1
                    if self.silence_count >= self.min_silence_frames:
                        self.speech_state = False

        return self.speech_state

    def reset(self):
        self.speech_state = False
        self.speech_count = 0
        self.silence_count = 0


def create_vad(config) -> object:
    if is_silero_available():
        try:
            return SileroVAD(
                threshold=config.threshold,
                sample_rate=16000,
                min_speech_ms=config.min_speech_ms,
                min_silence_ms=config.min_silence_ms,
            )
        except Exception as e:
            logger.warning(f"Silero VAD failed, falling back to energy VAD: {e}")

    return EnergyVAD(
        threshold=0.015,
        sample_rate=16000,
        min_speech_ms=config.min_speech_ms,
        min_silence_ms=config.min_silence_ms,
    )
