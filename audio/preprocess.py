import numpy as np
import logging
from typing import Optional

from config import config_manager

logger = logging.getLogger(__name__)

_noisereduce = None
_nr_checked = False


def _get_noisereduce():
    global _noisereduce, _nr_checked
    if _nr_checked:
        return _noisereduce
    _nr_checked = True
    try:
        import noisereduce as nr
        _noisereduce = nr
        return nr
    except ImportError:
        logger.warning("noisereduce not installed, noise reduction unavailable")
        return None


class AudioPreprocessor:
    def __init__(self):
        self.config = config_manager.audio
        self.sample_rate = self.config.sample_rate
        self.noise_profile: Optional[np.ndarray] = None

    def reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        nr = _get_noisereduce()
        if nr is None:
            return audio
        try:
            noise_clip = self.noise_profile
            if noise_clip is None:
                noise_samples = min(int(0.5 * self.sample_rate), len(audio))
                noise_clip = audio[:noise_samples]
            return nr.reduce_noise(
                y=audio,
                y_noise=noise_clip,
                sr=self.sample_rate,
                prop_decrease=self.config.noise_reduction_strength,
                stationary=False,
            )
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return audio

    def normalize(self, audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        rms = np.sqrt(np.mean(audio ** 2))
        if rms == 0:
            return audio
        current_db = 20 * np.log10(rms)
        gain = 10 ** ((target_db - current_db) / 20)
        return np.clip(audio * gain, -1.0, 1.0)

    def estimate_noise(self, audio: np.ndarray, duration: float = 0.5):
        n = int(duration * self.sample_rate)
        self.noise_profile = audio[:min(n, len(audio))]

    def process(self, audio: np.ndarray, denoise: bool = True) -> np.ndarray:
        if len(audio) == 0:
            return audio
        processed = audio.copy()
        if denoise:
            processed = self.reduce_noise(processed)
        return processed
