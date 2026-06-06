import os
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class SubtitleExporter:
    """Export subtitle history to SRT, VTT, or plain text format."""

    @staticmethod
    def export(history: List[Tuple[str, str, str]], filepath: str):
        """Export subtitles.

        Args:
            history: List of (original, translated, timestamp_str) tuples
            filepath: Output file path. Format determined by extension.
        """
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".srt":
            SubtitleExporter._export_srt(history, filepath)
        elif ext == ".vtt":
            SubtitleExporter._export_vtt(history, filepath)
        else:
            SubtitleExporter._export_txt(history, filepath)
        logger.info(f"Subtitles exported to {filepath}")

    @staticmethod
    def _export_srt(history, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            for i, (orig, trans, ts) in enumerate(history, 1):
                start = SubtitleExporter._ts_to_srt_time(ts, i - 1)
                end = SubtitleExporter._ts_to_srt_time(ts, i - 1, offset=3)
                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{orig}\n{trans}\n\n")

    @staticmethod
    def _export_vtt(history, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for i, (orig, trans, ts) in enumerate(history, 1):
                start = SubtitleExporter._ts_to_vtt_time(ts, i - 1)
                end = SubtitleExporter._ts_to_vtt_time(ts, i - 1, offset=3)
                f.write(f"{start} --> {end}\n")
                f.write(f"{orig}\n{trans}\n\n")

    @staticmethod
    def _export_txt(history, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            for orig, trans, ts in history:
                f.write(f"[{ts}]\n")
                f.write(f"  Original:    {orig}\n")
                f.write(f"  Translation: {trans}\n\n")

    @staticmethod
    def _ts_to_srt_time(ts_str: str, index: int, offset: int = 0) -> str:
        """Convert HH:MM:SS timestamp string to SRT format."""
        try:
            parts = ts_str.split(":")
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
            total_s = h * 3600 + m * 60 + s + offset
            h2 = total_s // 3600
            m2 = (total_s % 3600) // 60
            s2 = total_s % 60
            return f"{h2:02d}:{m2:02d}:{s2:02d},000"
        except Exception:
            seconds = index * 3 + offset
            h = seconds // 3600
            m = (seconds % 3600) // 60
            s = seconds % 60
            return f"{h:02d}:{m:02d}:{s:02d},000"

    @staticmethod
    def _ts_to_vtt_time(ts_str: str, index: int, offset: int = 0) -> str:
        srt_time = SubtitleExporter._ts_to_srt_time(ts_str, index, offset)
        return srt_time.replace(",", ".")
