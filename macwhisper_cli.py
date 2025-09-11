"""
A command-line tool to transcribe audio/video files using Whisper (stable-ts).
Adds:
 - Logging + optional progress bars
 - Configurable line width (with sensible defaults per language)
 - Multiple output formats (srt/vtt/txt/all) with default
 - Structured exit codes

Exit codes:
 0 - success
 1 - runtime error
 2 - usage error (invalid args, file not found, etc.)
 3 - dependency error (e.g., ffmpeg missing)
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# --- Third-party deps ---
# NOTE: These imports will fail fast if the packages are missing.
# We will still try to surface nicer error messages later for common cases.
import stable_whisper as whisper  # stable-ts enhanced whisper
import torch
from pydub import AudioSegment

# tqdm is optional; fall back gracefully
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - purely optional
    tqdm = None  # type: ignore

# ---------------------------
# Constants / Exit codes
# ---------------------------
EXIT_SUCCESS = 0
EXIT_RUNTIME_ERROR = 1
EXIT_USAGE_ERROR = 2
EXIT_DEPENDENCY_ERROR = 3

DEFAULT_MODEL = "medium"
DEFAULT_MAX_LINE_WIDTH = 42
DEFAULT_OUTPUT_FORMAT = "srt"  # choices: srt, vtt, txt, all
OUTPUT_FORMAT_CHOICES = ("srt", "vtt", "txt", "all")

# ---------------------------
# Logging
# ---------------------------
LOGGER = logging.getLogger("macwhisper_cli")


def setup_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    LOGGER.debug("Logging initialized at %s", level.upper())


# ---------------------------
# Utility helpers
# ---------------------------
def pick_device(cli_device: Optional[str]) -> str:
    if cli_device == "mps" and torch.backends.mps.is_available():
        return "mps"
    if cli_device == "cpu":
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _format_ts_srt(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1_000
    ms %= 1_000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_ts_vtt(seconds: float) -> str:
    # Same as SRT but with '.' separator
    ms = max(0, int(round(seconds * 1000)))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1_000
    ms %= 1_000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _is_cjk(lang_code: Optional[str]) -> bool:
    return (lang_code or "").lower() in {"zh", "ja", "zh-cn", "zh-tw", "zh-hk"}


def _wrap_lines(text: str, max_width: int, lang_code: Optional[str]) -> List[str]:
    """
    Language-aware wrapping:
    - CJK: wrap by characters
    - Others: wrap by words
    """
    text = (text or "").strip()
    if not text:
        return []

    lines: List[str] = []
    line = ""
    if _is_cjk(lang_code):
        tokens = list(text)
        joiner = ""
    else:
        tokens = text.split()
        joiner = " "

    for tok in tokens:
        if line and len(line) + len(joiner) + len(tok) > max_width:
            lines.append(line)
            line = tok
        else:
            if line:
                line = f"{line}{joiner}{tok}"
            else:
                line = tok

    if line:
        lines.append(line)
    return lines


# ---------------------------
# Core class
# ---------------------------
@dataclass
class WhisperTranscriber:
    model_name: str = DEFAULT_MODEL
    device: str = "mps"
    no_fp16: bool = False
    log_progress: bool = True

    def __post_init__(self) -> None:
        self.use_fp16 = self.device == "mps" and not self.no_fp16
        self.model = self._load_model()

    # ---- model ----
    def _load_model(self):
        LOGGER.info("Loading model '%s' on %s (fp16=%s)...", self.model_name, self.device, self.use_fp16)
        try:
            model = whisper.load_model(self.model_name, device=self.device)
            return model
        except Exception as e:
            LOGGER.error("Error loading model: %s", e)
            LOGGER.error(
                "Troubleshooting:\n"
                "  1) Close other memory-heavy apps and retry.\n"
                "  2) Try a smaller model (e.g., --model small).\n"
                "  3) On MPS, consider --no-fp16 if you see NaNs."
            )
            raise

    # ---- audio helpers ----
    def get_audio_duration(self, audio_path: str) -> float:
        audio = AudioSegment.from_file(audio_path)
        return len(audio) / 1000.0

    def cut_audio_to_remaining(self, audio_path: str, start_time: float, audio_duration: float) -> str:
        audio = AudioSegment.from_file(audio_path)
        remaining_audio = audio[int(start_time * 1000): int(audio_duration * 1000)]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            remaining_path = tmp.name
        remaining_audio.export(remaining_path, format="wav")
        return remaining_path

    # ---- main ops ----
    def transcribe(self, audio_path: str, **transcribe_options):
        LOGGER.info("Transcribing: %s", Path(audio_path).name)

        alignment_text_path = transcribe_options.get("align_from")
        if alignment_text_path:
            LOGGER.info("Running in ALIGNMENT mode (text: %s)", alignment_text_path)
            try:
                with open(alignment_text_path, "r", encoding="utf-8") as f:
                    alignment_text = f.read()
            except FileNotFoundError:
                LOGGER.error("Alignment file not found: %s", alignment_text_path)
                raise

            align_opts = dict(transcribe_options)
            align_opts.pop("align_from", None)
            return self.align(audio_path, alignment_text, **align_opts)

        # normal transcription
        return self.transcribe_stable(audio_path, **transcribe_options)

    def transcribe_stable(self, audio_path: str, **transcribe_options):
        verbose = transcribe_options.pop("verbose", False)
        transcribe_options["fp16"] = self.use_fp16
        LOGGER.debug("Transcribe options: %s", transcribe_options)
        result = self.model.transcribe(
            audio_path,
            verbose=verbose,
            condition_on_previous_text=True,
            **transcribe_options,
        )
        return result

    def align(self, audio_path: str, text: str, **transcribe_options):
        # Sanitize options not used in align
        transcribe_options.pop("initial_prompt", None)
        transcribe_options.pop("task", None)
        transcribe_options.pop("fp16", None)

        language = transcribe_options.pop("language", None)
        if language is None:
            raise ValueError("Alignment mode requires --language")

        if self.device == "mps":
            LOGGER.warning("Aligning on CPU due to MPS compatibility.")
            self.model.to("cpu")

        result = self.model.align(
            audio_path, text, language=language, ignore_compatibility=True, **transcribe_options
        )
        return result

    # ---- writers ----
    def write_srt(self, result: dict, path: Path, audio_duration: float, lang_code: Optional[str], max_width: int,
                  show_progress: bool = True):
        segs = result.get("segments", []) or []
        total = len(segs)
        iterator = segs

        pbar = None
        if tqdm and show_progress:
            pbar = tqdm(total=total, desc="Writing SRT", unit="seg")
        try:
            with open(path, "w", encoding="utf-8") as f:
                for idx, seg in enumerate(iterator, start=1):
                    start = float(seg["start"])
                    end = float(seg["end"])
                    text = (seg.get("text") or "").strip()

                    if start > audio_duration:
                        LOGGER.warning(
                            "Audio ended before text; stopping at %s", _format_ts_srt(start)
                        )
                        break

                    end = min(end, audio_duration)
                    if end <= start or not text:
                        if pbar:
                            pbar.update(1)
                        continue

                    start_s = _format_ts_srt(start)
                    end_s = _format_ts_srt(end)
                    lines = _wrap_lines(text, max_width=max_width, lang_code=lang_code)
                    f.write(f"{idx}\n{start_s} --> {end_s}\n" + "\n".join(lines) + "\n\n")
                    if pbar:
                        pbar.update(1)
            LOGGER.info("Saved SRT: %s", path.resolve())
        finally:
            if pbar:
                pbar.close()

    def write_vtt(self, result: dict, path: Path, audio_duration: float, lang_code: Optional[str], max_width: int,
                  show_progress: bool = True):
        segs = result.get("segments", []) or []
        total = len(segs)
        iterator = segs
        pbar = None
        if tqdm and show_progress:
            pbar = tqdm(total=total, desc="Writing VTT", unit="seg")

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("WEBVTT\n\n")
                for seg in iterator:
                    start = float(seg["start"])
                    end = float(seg["end"])
                    text = (seg.get("text") or "").strip()

                    if start > audio_duration:
                        LOGGER.warning("Audio ended before text; stopping at %s", _format_ts_vtt(start))
                        break

                    end = min(end, audio_duration)
                    if end <= start or not text:
                        if pbar:
                            pbar.update(1)
                        continue

                    start_s = _format_ts_vtt(start)
                    end_s = _format_ts_vtt(end)
                    lines = _wrap_lines(text, max_width=max_width, lang_code=lang_code)
                    f.write(f"{start_s} --> {end_s}\n" + "\n".join(lines) + "\n\n")

                    if pbar:
                        pbar.update(1)
            LOGGER.info("Saved VTT: %s", path.resolve())
        finally:
            if pbar:
                pbar.close()

    def write_txt(self, result: dict, path: Path):
        text = (result.get("text") or "").strip()
        with open(path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        LOGGER.info("Saved TXT: %s", path.resolve())


# ---------------------------
# CLI
# ---------------------------
def ensure_dependencies() -> None:
    # Basic FFmpeg presence check for pydub
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH. Install via Homebrew: brew install ffmpeg")


def reconstruct_result_if_needed(result, lang: Optional[str]) -> Optional[dict]:
    """
    Some align() variants can return a list of segments objects rather than a dict.
    We reconstruct a dict with 'text' and 'segments' if needed.
    """
    if result and not isinstance(result, dict):
        LOGGER.warning("Alignment returned segments list; reconstructing result.")
        try:
            reconstructed_segments = []
            sep = "" if _is_cjk(lang) else " "
            for seg in result:
                if hasattr(seg, "words") and seg.words:
                    seg_text = sep.join(w.word for w in seg.words).strip()
                else:
                    seg_text = (getattr(seg, "text", "") or "").strip()
                reconstructed_segments.append({"start": seg.start, "end": seg.end, "text": seg_text})
            full_text = sep.join(seg["text"] for seg in reconstructed_segments)
            return {"text": full_text, "segments": reconstructed_segments}
        except Exception as e:
            LOGGER.error("Failed to reconstruct alignment result: %s", e)
            return None
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe audio/video with Whisper (stable-ts) and write subtitles."
    )
    parser.add_argument("input_file", type=str, help="Input audio or video file path.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                        help="Whisper model.")
    parser.add_argument("--language", type=str, default=None,
                        help="Language code (e.g., zh, en). Required for alignment mode.")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"],
                        help="Task: transcribe (default) or translate to English.")
    parser.add_argument("--align_from", type=str, default=None,
                        help="Path to a text file for forced alignment.")
    parser.add_argument("--initial_prompt", type=str, default=None,
                        help="Optional initial prompt for the first window.")
    parser.add_argument("--no-fp16", action="store_true",
                        help="Force fp32 (disable fp16). Helps on some MPS setups.")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "mps"],
                        help="Device override. Auto-detect if omitted.")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save outputs.")
    # NEW: logging + progress
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity.")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable progress bars.")
    # NEW: line width + formats
    parser.add_argument("--max-line-width", type=int, default=DEFAULT_MAX_LINE_WIDTH,
                        help=f"Subtitle wrapping width (default {DEFAULT_MAX_LINE_WIDTH}).")
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
        transcriber = WhisperTranscriber(
            model_name=args.model,
            device=device,
            no_fp16=args.no_fp16,
            log_progress=not args.no_progress,
        )
    except Exception as e:
        LOGGER.error("Failed to initialize transcriber: %s", e)
        return EXIT_RUNTIME_ERROR

    # Do work
    start_time = time.time()
    try:
        transcribe_opts = {
            "language": args.language,
            "verbose": args.log_level == "DEBUG",  # Verbose model logs only in DEBUG
            "align_from": args.align_from,
            "task": args.task,
            "initial_prompt": args.initial_prompt,
        }

        # Duration for bounds checking in writers
        audio_duration = transcriber.get_audio_duration(str(input_path))

        result = transcriber.transcribe(str(input_path), **transcribe_opts)
        result = reconstruct_result_if_needed(result, args.language)

        if not result or not result.get("text"):
            LOGGER.error("Empty or invalid result.")
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
                transcriber.write_srt(
                    result, path, audio_duration=audio_duration,
                    lang_code=args.language, max_width=args.max_line_width,
                    show_progress=(not args.no_progress),
                )
            elif fmt == "vtt":
                transcriber.write_vtt(
                    result, path, audio_duration=audio_duration,
                    lang_code=args.language, max_width=args.max_line_width,
                    show_progress=(not args.no_progress),
                )
            elif fmt == "txt":
                transcriber.write_txt(result, path)

        elapsed = time.time() - start_time
        LOGGER.info("Done in %.2fs", elapsed)
        return EXIT_SUCCESS

    except Exception as e:
        LOGGER.error("Unhandled error: %s", e)
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    sys.exit(main())