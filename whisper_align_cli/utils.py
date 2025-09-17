"""
Whisper Align - Utility functions module
"""

import logging
import shutil
from typing import Optional

import torch

# ---------------------------
# Logging
# ---------------------------
LOGGER = logging.getLogger("macwhisper_align")


def setup_logging(level: str) -> None:
    """Setup logging configuration"""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    LOGGER.debug("Logging initialized at %s", level.upper())


# ---------------------------
# Device management
# ---------------------------
def pick_device(cli_device: Optional[str]) -> str:
    """Pick the appropriate device for model execution"""
    if cli_device == "mps" and torch.backends.mps.is_available():
        return "mps"
    if cli_device == "cpu":
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------
# Language utilities
# ---------------------------
def is_cjk(lang_code: str) -> bool:
    """Check if language code is CJK (Chinese, Japanese, Korean)"""
    cjk_codes = {"zh", "ja", "ko", "zh-cn", "zh-tw", "zh-hk", "kr"}
    return lang_code.lower() in cjk_codes


def remove_cjk_spaces(text: str) -> str:
    """Remove spaces from CJK text"""
    if not text:
        return text or ""
    return text.replace(' ', '')


# ---------------------------
# Text processing
# ---------------------------
def format_timestamp_srt(seconds: float) -> str:
    """Format timestamp for SRT format"""
    ms = max(0, int(round(seconds * 1000)))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1_000
    ms %= 1_000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Format timestamp for VTT format"""
    # Same as SRT but with '.' separator
    ms = max(0, int(round(seconds * 1000)))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1_000
    ms %= 1_000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def preprocess_alignment_text(text: str) -> str:
    """
    Preprocess text for alignment mode:
    - Add sentence-ending punctuation to lines that don't have it
    - This helps stable-whisper better segment the text
    """
    if not text:
        return ""

    lines = text.splitlines()
    processed_lines = []
    sentence_endings = {'.', '!', '?', '。', '！', '？', '…'}

    for line in lines:
        stripped = line.rstrip()
        if stripped:  # Only process non-empty lines
            # If line doesn't end with sentence-ending punctuation, add a period
            if stripped[-1] not in sentence_endings:
                # For CJK text, add ideographic full stop (。)
                if any('\u4e00' <= char <= '\u9fff' for char in stripped):
                    stripped += '。'
                else:
                    stripped += '.'
            processed_lines.append(stripped)
        else:
            # Preserve empty lines
            processed_lines.append(line)

    return '\n'.join(processed_lines)


def clean_alignment_text(text: str, original_text: str, is_cjk: bool) -> str:
    """
    Clean up alignment text by removing added punctuation and handling spaces.
    """
    if not text or not original_text:
        return text or ""

    processed_lines = text.splitlines()
    original_lines = original_text.splitlines()
    added_endings = {'.', '。'}
    cleaned_lines = []

    for processed_line, original_line in zip(processed_lines, original_lines):
        processed_stripped = processed_line.rstrip()
        original_stripped = original_line.rstrip()

        # If the original line was empty, keep the processed line as is
        if not original_stripped:
            cleaned_lines.append(processed_line)
            continue

        # If we added punctuation, remove it
        if (processed_stripped and original_stripped and
            processed_stripped[-1] in added_endings and
            (not original_stripped or original_stripped[-1] not in added_endings)):
            # Remove the last character (the added punctuation)
            cleaned_line = processed_stripped[:-1]
            # Preserve original trailing whitespace
            cleaned_line_with_spaces = cleaned_line + (processed_line[len(processed_stripped):])
        else:
            cleaned_line_with_spaces = processed_line

        # For CJK languages, remove all spaces
        if is_cjk:
            cleaned_lines.append(remove_cjk_spaces(cleaned_line_with_spaces))
        else:
            cleaned_lines.append(cleaned_line_with_spaces)

    # Handle case where processed text has more lines than original
    if len(processed_lines) > len(original_lines):
        extra_lines = processed_lines[len(original_lines):]
        for line in extra_lines:
            if is_cjk:
                cleaned_lines.append(remove_cjk_spaces(line))
            else:
                cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)