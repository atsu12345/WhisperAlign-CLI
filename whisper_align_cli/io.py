"""
Whisper Align - I/O operations module
"""

import logging
from pathlib import Path
from typing import Optional

from .utils import LOGGER, is_cjk, remove_cjk_spaces, format_timestamp_srt, format_timestamp_vtt

# tqdm is optional; fall back gracefully
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


class OutputWriter:
    """Handles writing alignment results to various formats"""
    
    def __init__(self, log_progress: bool = True):
        self.log_progress = log_progress

    def _write_with_progress(self, segments: list, writer_func, path: Path, 
                           audio_duration: float, language: str, format_name: str) -> None:
        """Helper method to write output with progress bar"""
        total = len(segments)
        pbar = None
        if tqdm and self.log_progress:
            pbar = tqdm(total=total, desc=f"Writing {format_name}", unit="seg")
        
        try:
            writer_func(segments, path, audio_duration, language, pbar)
            LOGGER.info("Saved %s: %s", format_name, path.resolve())
        finally:
            if pbar:
                pbar.close()

    def write_srt(self, result: dict, path: Path, audio_duration: float, language: str):
        """Write result to SRT format"""
        segs = result.get("segments", []) or []
        
        def _write_srt_segments(segments, path, audio_duration, language, pbar):
            with open(path, "w", encoding="utf-8") as f:
                for idx, seg in enumerate(segments, start=1):
                    start = float(seg["start"])
                    end = float(seg["end"])
                    text = (seg.get("text") or "").strip()

                    # Remove spaces for CJK languages
                    if is_cjk(language):
                        text = remove_cjk_spaces(text)

                    if start > audio_duration:
                        LOGGER.warning(
                            "Audio ended before text; stopping at %s", format_timestamp_srt(start)
                        )
                        break

                    end = min(end, audio_duration)
                    if end <= start or not text:
                        if pbar:
                            pbar.update(1)
                        continue

                    start_s = format_timestamp_srt(start)
                    end_s = format_timestamp_srt(end)
                    f.write(f"{idx}\n{start_s} --> {end_s}\n{text}\n\n")
                    if pbar:
                        pbar.update(1)

        self._write_with_progress(segs, _write_srt_segments, path, audio_duration, language, "SRT")

    def write_vtt(self, result: dict, path: Path, audio_duration: float, language: str):
        """Write result to VTT format"""
        segs = result.get("segments", []) or []
        
        def _write_vtt_segments(segments, path, audio_duration, language, pbar):
            with open(path, "w", encoding="utf-8") as f:
                f.write("WEBVTT\n\n")
                for seg in segments:
                    start = float(seg["start"])
                    end = float(seg["end"])
                    text = (seg.get("text") or "").strip()

                    # Remove spaces for CJK languages
                    if is_cjk(language):
                        text = remove_cjk_spaces(text)

                    if start > audio_duration:
                        LOGGER.warning("Audio ended before text; stopping at %s", format_timestamp_vtt(start))
                        break

                    end = min(end, audio_duration)
                    if end <= start or not text:
                        if pbar:
                            pbar.update(1)
                        continue

                    start_s = format_timestamp_vtt(start)
                    end_s = format_timestamp_vtt(end)
                    f.write(f"{start_s} --> {end_s}\n{text}\n\n")

                    if pbar:
                        pbar.update(1)

        self._write_with_progress(segs, _write_vtt_segments, path, audio_duration, language, "VTT")

    def write_txt(self, result: dict, path: Path, language: str):
        """Write result to TXT format"""
        text = (result.get("text") or "").strip()
        # Remove spaces for CJK languages
        if is_cjk(language):
            text = remove_cjk_spaces(text)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        LOGGER.info("Saved TXT: %s", path.resolve())