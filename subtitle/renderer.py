import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class SubtitleLine:
    original: str
    translated: str
    timestamp: float
    display_until: float = 0.0


class SubtitleRenderer:
    """Manages the active subtitle lines for display."""

    def __init__(self, max_lines: int = 3, display_duration: float = 8.0):
        self.max_lines = max_lines
        self.display_duration = display_duration
        self.lines: deque[SubtitleLine] = deque(maxlen=max_lines)
        self.all_lines: List[SubtitleLine] = []

    def add_line(self, original: str, translated: str):
        now = time.time()
        line = SubtitleLine(
            original=original,
            translated=translated,
            timestamp=now,
            display_until=now + self.display_duration,
        )
        self.lines.append(line)
        self.all_lines.append(line)

    def get_active_lines(self) -> List[SubtitleLine]:
        now = time.time()
        active = [l for l in self.lines if l.display_until > now]
        return active

    def get_all_lines(self) -> List[SubtitleLine]:
        return self.all_lines

    def clear(self):
        self.lines.clear()
        self.all_lines.clear()

    def correct_line(self, index: int, new_translated: str) -> Optional[str]:
        if 0 <= index < len(self.all_lines):
            old = self.all_lines[index].translated
            self.all_lines[index].translated = new_translated
            return old
        return None
