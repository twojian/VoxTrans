import numpy as np
import threading
import queue
import time
import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass

from config import config_manager

logger = logging.getLogger(__name__)

_pyaudio = None


def _get_pyaudio():
    global _pyaudio
    if _pyaudio is None:
        import pyaudio
        _pyaudio = pyaudio
    return _pyaudio


def _get_pyaudio_instance():
    return _get_pyaudio().PyAudio()


_scipy_signal = None
_scipy_checked = False


def _get_scipy_signal():
    global _scipy_signal, _scipy_checked
    if _scipy_checked:
        return _scipy_signal
    _scipy_checked = True
    try:
        from scipy import signal
        _scipy_signal = signal
        return _scipy_signal
    except ImportError:
        return None


@dataclass
class AudioChunk:
    data: np.ndarray
    sample_rate: int
    timestamp: float
    duration: float

    @property
    def samples(self) -> int:
        return len(self.data)


class Resampler:
    TARGET_SAMPLE_RATE = 16000

    @staticmethod
    def resample(audio_data: np.ndarray, orig_sr: int,
                 target_sr: int = 16000) -> Tuple[np.ndarray, float]:
        if orig_sr == target_sr:
            return audio_data, 1.0
        if len(audio_data) == 0:
            return audio_data, 1.0

        ratio = target_sr / orig_sr
        scipy_signal = _get_scipy_signal()

        if scipy_signal is not None:
            try:
                resampled = scipy_signal.resample(audio_data, int(len(audio_data) * ratio))
                return resampled.astype(np.float32), ratio
            except Exception:
                pass

        orig_indices = np.arange(len(audio_data))
        target_indices = np.linspace(0, len(audio_data) - 1, int(len(audio_data) * ratio))
        resampled = np.interp(target_indices, orig_indices, audio_data)
        return resampled.astype(np.float32), ratio


SYSTEM_AUDIO_KEYWORDS = [
    "stereo mix", "立体声混音", "立体声混合",
    "what u hear", "您听到的声音", "loopback", "回环",
    "系统音频", "系统声音", "wave out",
    "cable input", "cable output", "vb-cable", "vb cable",
]

MIC_KEYWORDS = [
    "microphone", "麦克风", "mic ", "headset", "耳机",
    "array", "阵列", "internal mic", "内置麦克风",
]


class AudioCapture:
    COMMON_SAMPLE_RATES = [16000, 44100, 48000, 22050, 8000]

    def __init__(self):
        self.config = config_manager.audio
        self.p = None
        self.stream = None
        self.is_recording = False
        self.audio_queue: queue.Queue = queue.Queue(maxsize=100)
        self.input_device_index: Optional[int] = None
        self.actual_sample_rate: int = self.config.sample_rate
        self.actual_channels: int = 1
        self._total_chunks = 0

    def list_devices(self) -> List[dict]:
        if self.p is None:
            self.p = _get_pyaudio_instance()
        devices = []
        for i in range(self.p.get_device_count()):
            try:
                info = self.p.get_device_info_by_index(i)
                if info["maxInputChannels"] > 0:
                    name = info["name"]
                    name_lower = name.lower()
                    is_system = any(kw in name_lower for kw in SYSTEM_AUDIO_KEYWORDS)
                    is_mic = any(kw in name_lower for kw in MIC_KEYWORDS)
                    devices.append({
                        "index": i,
                        "name": name,
                        "channels": info["maxInputChannels"],
                        "default_sample_rate": int(info["defaultSampleRate"]),
                        "is_system_audio": is_system,
                        "is_microphone": is_mic,
                    })
            except Exception:
                continue
        return devices

    def get_default_device(self) -> Optional[int]:
        if self.p is None:
            self.p = _get_pyaudio_instance()
        try:
            return self.p.get_default_input_device_info()["index"]
        except Exception:
            return None

    def set_device(self, device_index: int):
        self.input_device_index = device_index
        if self.p is None:
            self.p = _get_pyaudio_instance()
        try:
            info = self.p.get_device_info_by_index(device_index)
            logger.info(f"Selected device: [{device_index}] {info['name']}")
        except Exception as e:
            logger.warning(f"Failed to get device info: {e}")

    def start(self):
        if self.is_recording:
            return
        if self.p is None:
            self.p = _get_pyaudio_instance()
        if self.input_device_index is None:
            self.input_device_index = self.get_default_device()
            if self.input_device_index is None:
                raise RuntimeError("No audio input device available")

        info = self.p.get_device_info_by_index(self.input_device_index)
        channels = min(2, int(info["maxInputChannels"]))
        success = self._try_open_stream(info, channels)
        if not success:
            raise RuntimeError(f"Cannot open audio device [{self.input_device_index}] {info['name']}")

        self.is_recording = True
        self.stream.start_stream()
        self._total_chunks = 0
        logger.info("Audio capture started")

    def _try_open_stream(self, device_info, preferred_channels) -> bool:
        pyaudio = _get_pyaudio()
        default_sr = int(device_info["defaultSampleRate"])
        max_ch = int(device_info["maxInputChannels"])

        sample_rates = list(dict.fromkeys(
            [self.config.sample_rate, default_sr, 44100, 48000, 16000, 8000]
        ))
        channels_list = list(dict.fromkeys([1, preferred_channels, min(max_ch, 2)]))
        if not channels_list:
            channels_list = [1]

        for sr in sample_rates:
            for ch in channels_list:
                for chunk in [1024, 2048, 512]:
                    try:
                        self.stream = self.p.open(
                            format=pyaudio.paInt16,
                            channels=ch,
                            rate=sr,
                            input=True,
                            input_device_index=self.input_device_index,
                            frames_per_buffer=chunk,
                            stream_callback=self._callback,
                        )
                        self.actual_sample_rate = sr
                        self.actual_channels = ch
                        logger.info(f"Opened stream: sr={sr}, ch={ch}, chunk={chunk}")
                        return True
                    except Exception:
                        continue
        return False

    def _callback(self, in_data, frame_count, time_info, status):
        try:
            audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0

            if self.actual_channels > 1:
                try:
                    audio_data = audio_data.reshape(-1, self.actual_channels).mean(axis=1)
                except Exception:
                    audio_data = audio_data[::self.actual_channels]

            if self.actual_sample_rate != Resampler.TARGET_SAMPLE_RATE:
                audio_data, _ = Resampler.resample(
                    audio_data, self.actual_sample_rate, Resampler.TARGET_SAMPLE_RATE
                )

            if self.config.volume_gain > 1.0:
                audio_data = np.clip(audio_data * self.config.volume_gain, -1.0, 1.0)

            duration = len(audio_data) / Resampler.TARGET_SAMPLE_RATE
            chunk = AudioChunk(
                data=audio_data,
                sample_rate=Resampler.TARGET_SAMPLE_RATE,
                timestamp=time.time(),
                duration=duration,
            )

            if not self.audio_queue.full():
                self.audio_queue.put_nowait(chunk)
            else:
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(chunk)
                except Exception:
                    pass

            self._total_chunks += 1
        except Exception as e:
            logger.error(f"Audio callback error: {e}")

        return (in_data, _get_pyaudio().paContinue)

    def read_chunk(self, timeout: Optional[float] = None) -> Optional[AudioChunk]:
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def read_all_available(self) -> List[AudioChunk]:
        chunks = []
        while not self.audio_queue.empty():
            try:
                chunks.append(self.audio_queue.get_nowait())
            except queue.Empty:
                break
        return chunks

    def stop(self):
        self.is_recording = False
        if self.stream is not None:
            stream = self.stream
            self.stream = None
            try:
                stream.stop_stream()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Exception:
                break
        logger.info(f"Audio capture stopped, total chunks: {self._total_chunks}")

    def close(self):
        self.stop()
        if self.p is not None:
            try:
                self.p.terminate()
            except Exception:
                pass
            self.p = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.close()


class AudioBuffer:
    def __init__(self, sample_rate: int = 16000, buffer_duration: float = 30.0):
        self.sample_rate = sample_rate
        self.buffer_samples = int(buffer_duration * sample_rate)
        self.buffer = np.zeros(self.buffer_samples, dtype=np.float32)
        self.write_pos = 0
        self.lock = threading.Lock()

    def add_chunk(self, chunk: AudioChunk):
        data = chunk.data
        with self.lock:
            if self.write_pos + len(data) > self.buffer_samples:
                shift = self.write_pos + len(data) - self.buffer_samples
                self.buffer[:-shift] = self.buffer[shift:]
                self.write_pos -= shift
            self.buffer[self.write_pos:self.write_pos + len(data)] = data
            self.write_pos += len(data)

    def get_buffer(self) -> np.ndarray:
        with self.lock:
            return self.buffer[:self.write_pos].copy()

    def get_duration(self) -> float:
        with self.lock:
            return self.write_pos / self.sample_rate

    def clear(self):
        with self.lock:
            self.buffer.fill(0)
            self.write_pos = 0
