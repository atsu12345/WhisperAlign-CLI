"""
Whisper Align - CLI interface module
"""

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

from pydub import AudioSegment

from .core import WhisperAligner, DEFAULT_MODEL
from .io import OutputWriter
from .utils import LOGGER, setup_logging, pick_device, is_cjk

# ---------------------------
# Constants / Exit codes
# ---------------------------
EXIT_SUCCESS = 0
EXIT_RUNTIME_ERROR = 1
EXIT_USAGE_ERROR = 2
EXIT_DEPENDENCY_ERROR = 3

DEFAULT_OUTPUT_FORMAT = "srt"  # choices: srt, vtt, txt, all
OUTPUT_FORMAT_CHOICES = ("srt", "vtt", "txt", "all")


# ---------------------------
# Dependencies
# ---------------------------
def ensure_dependencies() -> None:
    """Check for required dependencies"""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH. Install via Homebrew: brew install ffmpeg")


# ---------------------------
# Audio utilities
# ---------------------------
def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds"""
    audio = AudioSegment.from_file(audio_path)
    return len(audio) / 1000.0


# ---------------------------
# Main CLI
# ---------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Perform forced alignment of text with audio using Whisper (stable-ts)."
    )
    parser.add_argument("input_file", type=str, help="Input audio or video file path.")
    parser.add_argument("text_file", type=str, help="Path to reference text file for alignment.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                        help="Whisper model.")
    parser.add_argument("--language", type=str, required=True,
                        help="Language code (e.g., zh, ja, en). Required for alignment.")
    parser.add_argument("--no-fp16", action="store_true",
                        help="Force fp32 (disable fp16). Helps on some MPS setups.")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "mps"],
                        help="Device override. Auto-detect if omitted.")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save outputs.")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity.")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable progress bars.")
    parser.add_argument("--output-format", type=str, default=DEFAULT_OUTPUT_FORMAT,
                        choices=OUTPUT_FORMAT_CHOICES,
                        help="Output format: srt (default), vtt, txt, or all.")

    args = parser.parse_args(argv)

    # Setup logging
    setup_logging(args.log_level)

    # Basic arg validation
    input_path = Path(args.input_file)
    if not input_path.exists():
        LOGGER.error("Input file not found: %s", input_path)
        return EXIT_USAGE_ERROR
        
    text_path = Path(args.text_file)
    if not text_path.exists():
        LOGGER.error("Reference text file not found: %s", text_path)
        return EXIT_USAGE_ERROR

    # Dependencies
    try:
        ensure_dependencies()
    except RuntimeError as dep_err:
        LOGGER.error(str(dep_err))
        return EXIT_DEPENDENCY_ERROR

    # Device
    device = pick_device(args.device)
    LOGGER.info("Using device: %s", device)

    # Init
    try:
        aligner = WhisperAligner(
            model_name=args.model,
            device=device,
            no_fp16=args.no_fp16,
            log_progress=not args.no_progress,
        )
        writer = OutputWriter(log_progress=not args.no_progress)
    except Exception as e:
        LOGGER.error("Failed to initialize aligner: %s", e)
        return EXIT_RUNTIME_ERROR

    # Do work
    start_time = time.time()
    try:
        # Duration for bounds checking in writers
        audio_duration = get_audio_duration(str(input_path))

        # Perform alignment
        result = aligner.align(str(input_path), str(text_path), args.language)

        # Check if result is valid
        if not result or not result.get("text"):
            LOGGER.error("Empty or invalid alignment result.")
            return EXIT_RUNTIME_ERROR

        LOGGER.debug("Full text preview (first 400 chars): %s", result["text"][:400])

        # Output
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        base = input_path.stem

        to_write = []
        if args.output_format in ("srt", "all"):
            to_write.append(("srt", out_dir / f"{base}.srt"))
        if args.output_format in ("vtt", "all"):
            to_write.append(("vtt", out_dir / f"{base}.vtt"))
        if args.output_format in ("txt", "all"):
            to_write.append(("txt", out_dir / f"{base}.txt"))

        for fmt, path in to_write:
            if fmt == "srt":
                writer.write_srt(
                    result, path, audio_duration=audio_duration,
                    language=args.language
                )
            elif fmt == "vtt":
                writer.write_vtt(
                    result, path, audio_duration=audio_duration,
                    language=args.language
                )
            elif fmt == "txt":
                writer.write_txt(result, path, language=args.language)

        elapsed = time.time() - start_time
        LOGGER.info("Alignment completed in %.2fs", elapsed)
        return EXIT_SUCCESS

    except Exception as e:
        LOGGER.error("Unhandled error: %s", e)
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    sys.exit(main())