"""
MacWhisper Align - A command-line tool for forced alignment of text with audio using Whisper (stable-ts).

This package provides:
- Core alignment functionality (core.py)
- Utility functions (utils.py)
- I/O operations (io.py)
- CLI interface (cli.py)
"""

from .core import WhisperAligner
from .io import OutputWriter
from .utils import (
    setup_logging,
    pick_device,
    is_cjk,
    remove_cjk_spaces,
    format_timestamp_srt,
    format_timestamp_vtt,
    preprocess_alignment_text,
    clean_alignment_text
)

__version__ = "1.0.0"
__author__ = "Rael"