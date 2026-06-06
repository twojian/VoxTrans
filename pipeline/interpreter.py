import time
import logging
import threading
import queue
import numpy as np
from typing import Optional, List

from PyQt6.QtCore import QObject, pyqtSignal

from config import config_manager
from audio.capture import AudioCapture, AudioChunk
from audio.vad import create_vad
from asr.base import create_asr, ASRResult
from translation.base import create_translator, TranslationResult
from correction.context_corrector import ContextCorrector

logger = logging.getLogger(__name__)


class InterpreterPipeline(QObject):
    """Core simultaneous interpretation pipeline.

    Orchestrates: AudioCapture → VAD → ASR → Translation → Correction
    Uses dedicated threads for audio processing and translation to avoid
    blocking the UI thread.
    """

    # Signals to UI
    subtitle_updated = pyqtSignal(str, str)       # (original, translated)
    correction_applied = pyqtSignal(int, str, str) # (line_index, old, new)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    audio_level = pyqtSignal(float)                # current audio RMS level

    def __init__(self):
        super().__init__()
        self.config = config_manager.config
        self.capture: Optional[AudioCapture] = None
        self.vad = None
        self.asr = None
        self.translator = None
        self.corrector = ContextCorrector()

        self._running = False
        self._paused = False
        self._audio_thread: Optional[threading.Thread] = None
        self._translate_thread: Optional[threading.Thread] = None
        self._asr_queue: queue.Queue = queue.Queue(maxsize=20)
        self._translate_queue: queue.Queue = queue.Queue(maxsize=20)

        # Accumulate audio chunks until we have enough for ASR
        self._audio_buffer = np.array([], dtype=np.float32)
        self._min_audio_seconds = 1.5
        self._max_audio_seconds = 10.0
        self._silence_timeout = 0.8

    def set_device(self, device_index: int):
        if self.capture is None:
            self.capture = AudioCapture()
        self.capture.set_device(device_index)

    def start(self, source_lang: str = "auto", target_lang: str = "zh"):
        if self._running:
            return

        self.status_changed.emit("Initializing...")

        try:
            # Init audio capture
            if self.capture is None:
                self.capture = AudioCapture()
            self.capture.start()

            # Init VAD
            self.vad = create_vad(self.config.vad)

            # Init ASR
            self.status_changed.emit("Loading ASR model...")
            self.asr = create_asr(self.config.asr)

            # Init translator
            self.status_changed.emit("Initializing translator...")
            self.translator = create_translator(self.config.translation)

            self._source_lang = source_lang
            self._target_lang = target_lang
            self._running = True
            self._paused = False

            # Start worker threads
            self._audio_thread = threading.Thread(
                target=self._audio_loop, daemon=True, name="AudioLoop"
            )
            self._translate_thread = threading.Thread(
                target=self._translate_loop, daemon=True, name="TranslateLoop"
            )
            self._audio_thread.start()
            self._translate_thread.start()

            self.status_changed.emit("Running")
            logger.info("Pipeline started")

        except Exception as e:
            logger.error(f"Pipeline start failed: {e}")
            self.error_occurred.emit(str(e))
            self.stop()

    def stop(self):
        self._running = False

        if self.capture:
            self.capture.stop()

        # Drain queues to unblock threads
        for q in (self._asr_queue, self._translate_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        if self._audio_thread and self._audio_thread.is_alive():
            self._audio_thread.join(timeout=3)
        if self._translate_thread and self._translate_thread.is_alive():
            self._translate_thread.join(timeout=3)

        self._audio_buffer = np.array([], dtype=np.float32)
        self.status_changed.emit("Stopped")
        logger.info("Pipeline stopped")

    def pause(self):
        self._paused = True
        self.status_changed.emit("Paused")

    def resume(self):
        self._paused = False
        self.status_changed.emit("Running")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused

    def _audio_loop(self):
        """Audio capture → VAD → ASR thread."""
        last_speech_time = time.time()

        while self._running:
            if self._paused:
                time.sleep(0.1)
                continue

            chunk = self.capture.read_chunk(timeout=0.1)
            if chunk is None:
                # Check silence timeout
                if (len(self._audio_buffer) > 0 and
                        time.time() - last_speech_time > self._silence_timeout):
                    self._flush_buffer()
                continue

            # Emit audio level
            rms = float(np.sqrt(np.mean(chunk.data ** 2)))
            self.audio_level.emit(rms)

            # VAD check
            is_speech = self.vad.process_chunk(chunk.data)

            if is_speech:
                last_speech_time = time.time()
                self._audio_buffer = np.concatenate([self._audio_buffer, chunk.data])

                # If buffer exceeds max duration, flush
                buf_duration = len(self._audio_buffer) / 16000
                if buf_duration >= self._max_audio_seconds:
                    self._flush_buffer()
            else:
                # Silence detected
                if len(self._audio_buffer) > 0:
                    buf_duration = len(self._audio_buffer) / 16000
                    if buf_duration >= self._min_audio_seconds:
                        self._flush_buffer()
                    elif time.time() - last_speech_time > self._silence_timeout:
                        self._flush_buffer()

    def _flush_buffer(self):
        """Send accumulated audio to ASR."""
        if len(self._audio_buffer) == 0:
            return

        buf_duration = len(self._audio_buffer) / 16000
        if buf_duration < 0.3:
            self._audio_buffer = np.array([], dtype=np.float32)
            return

        audio = self._audio_buffer.copy()
        self._audio_buffer = np.array([], dtype=np.float32)

        try:
            self._asr_queue.put_nowait(audio)
        except queue.Full:
            logger.warning("ASR queue full, dropping audio segment")

        # Run ASR in this thread to keep it sequential
        try:
            result = self.asr.recognize(audio, language=self._source_lang)
            if result.text.strip():
                self._translate_queue.put_nowait(result)
        except Exception as e:
            logger.error(f"ASR error: {e}")

    def _translate_loop(self):
        """Translation thread: ASR result → Translation → UI signal."""
        while self._running:
            try:
                asr_result = self._translate_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if self._paused:
                continue

            try:
                context = self.corrector.get_context_texts()
                tr_result = self.translator.translate(
                    asr_result.text,
                    source_lang=asr_result.language or self._source_lang,
                    target_lang=self._target_lang,
                    context=context,
                )

                # Check consistency
                corrected = self.corrector.check_consistency(
                    asr_result.text, tr_result.translated_text
                )
                if corrected:
                    tr_result.translated_text = corrected

                # Add to history
                self.corrector.add_entry(
                    asr_result.text, tr_result.translated_text, time.time()
                )

                # Emit to UI
                self.subtitle_updated.emit(
                    asr_result.text, tr_result.translated_text
                )

            except Exception as e:
                logger.error(f"Translation error: {e}")
                self.subtitle_updated.emit(asr_result.text, f"[Error: {e}]")

    def close(self):
        self.stop()
        if self.capture:
            self.capture.close()
            self.capture = None
        if self.asr:
            self.asr.close()
            self.asr = None
        if self.translator:
            self.translator.close()
            self.translator = None
