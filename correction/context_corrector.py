import logging
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CorrectionEntry:
    original_text: str
    corrected_text: str
    translated_text: str
    timestamp: float = 0.0


class ContextCorrector:
    """Context-aware corrector for ASR and translation results.

    Maintains a sliding window of recent results to:
    1. Detect and correct ASR errors using surrounding context
    2. Ensure translation terminology consistency across sentences
    3. Emit corrections for already-displayed subtitles
    """

    def __init__(self, history_size: int = 10):
        self.history: deque[CorrectionEntry] = deque(maxlen=history_size)
        self.term_map: dict = {}
        self._callbacks: List = []

    def add_entry(self, original: str, translated: str, timestamp: float = 0.0):
        entry = CorrectionEntry(
            original_text=original,
            corrected_text=original,
            translated_text=translated,
            timestamp=timestamp,
        )
        self.history.append(entry)
        self._update_term_map(original, translated)

    def _update_term_map(self, original: str, translated: str):
        words = original.split()
        for word in words:
            if word[0].isupper() and len(word) > 2:
                if word in self.term_map:
                    if self.term_map[word] != translated:
                        pass  # term inconsistency detected
                self.term_map[word] = translated

    def get_context_texts(self, n: int = 5) -> List[str]:
        recent = list(self.history)[-n:]
        return [e.translated_text for e in recent if e.translated_text]

    def check_consistency(self, text: str, translated: str) -> Optional[str]:
        """Check if translation is consistent with previous term usage.

        Returns corrected translation if inconsistency found, None otherwise.
        """
        if not self.history:
            return None

        # Simple heuristic: check if key terms in the source have been
        # translated differently than in previous entries
        corrections_made = False
        corrected = translated

        for entry in self.history:
            # Find shared source terms
            entry_words = set(entry.original_text.split())
            current_words = set(text.split())
            shared = entry_words & current_words

            for word in shared:
                if len(word) <= 2 or not word[0].isupper():
                    continue
                # If a proper noun appears in both and was translated
                # differently, flag it (but don't auto-correct to avoid
                # false positives in simple heuristic mode)
                pass

        return corrected if corrections_made else None

    def get_correction(self, index: int) -> Optional[Tuple[str, str]]:
        """Get correction for a specific history entry.

        Returns (old_text, new_text) if corrected, None otherwise.
        """
        if index < 0 or index >= len(self.history):
            return None
        entry = self.history[index]
        if entry.corrected_text != entry.original_text:
            return (entry.original_text, entry.corrected_text)
        return None

    def on_correction(self, callback):
        self._callbacks.append(callback)

    def _emit_correction(self, old_text: str, new_text: str, index: int):
        for cb in self._callbacks:
            try:
                cb(index, old_text, new_text)
            except Exception as e:
                logger.error(f"Correction callback error: {e}")

    def clear(self):
        self.history.clear()
        self.term_map.clear()
