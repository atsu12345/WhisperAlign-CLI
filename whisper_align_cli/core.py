"""
Whisper Align - Core alignment functionality
"""

import logging
from dataclasses import dataclass
from typing import Optional

import stable_whisper as whisper

from .utils import LOGGER, is_cjk, remove_cjk_spaces, preprocess_alignment_text, clean_alignment_text

# ---------------------------
# Constants
# ---------------------------
DEFAULT_MODEL = "medium"


@dataclass
class WhisperAligner:
    """Whisper-based audio-text aligner"""
    model_name: str = DEFAULT_MODEL
    device: str = "mps"
    no_fp16: bool = False
    log_progress: bool = True

    def __post_init__(self) -> None:
        self.use_fp16 = self.device == "mps" and not self.no_fp16
        self.model = self._load_model()

    def _load_model(self):
        """Load the Whisper model"""
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

    def align(self, audio_path: str, text_path: str, language: str):
        """
        Perform forced alignment of text with audio.
        
        Args:
            audio_path: Path to audio file
            text_path: Path to reference text file
            language: Language code (required for alignment)
            
        Returns:
            Alignment result with timestamps
        """
        LOGGER.info("Running forced alignment: audio=%s, text=%s", 
                   audio_path.split('/')[-1], text_path.split('/')[-1])
        
        # Read reference text
        try:
            with open(text_path, "r", encoding="utf-8") as f:
                alignment_text = f.read()
        except FileNotFoundError:
            LOGGER.error("Reference text file not found: %s", text_path)
            raise

        # Store original text for cleanup later
        original_alignment_text = alignment_text

        # Preprocess alignment text to add sentence-ending punctuation
        alignment_text = preprocess_alignment_text(alignment_text)

        # Align on CPU for stability
        if self.device == "mps":
            LOGGER.warning("Aligning on CPU due to MPS compatibility.")
            self.model.to("cpu")

        # Perform alignment
        result = self.model.align(
            audio_path, alignment_text, language=language, ignore_compatibility=True
        )
        
        # Clean up the result segments using the original text
        cleaned_result = self._clean_result_segments(result, original_alignment_text, language)

        return cleaned_result

    def _clean_result_segments(self, result, original_text: str, language: str) -> dict:
        """
        Clean up the result segments by removing added punctuation and CJK spaces.
        """
        # Handle different result formats
        if not result:
            return {}

        # If result is a list of segments, convert to dict
        if not isinstance(result, dict):
            LOGGER.warning("Alignment returned segments list; reconstructing result.")
            try:
                reconstructed_segments = []
                is_cjk_lang = is_cjk(language)
                sep = "" if is_cjk_lang else " "
                
                for seg in result:
                    if hasattr(seg, "words") and seg.words:
                        seg_text = sep.join(w.word for w in seg.words).strip()
                    else:
                        seg_text = (getattr(seg, "text", "") or "").strip()
                    reconstructed_segments.append({"start": seg.start, "end": seg.end, "text": seg_text})
                
                full_text = sep.join(seg["text"] for seg in reconstructed_segments)
                result = {"text": full_text, "segments": reconstructed_segments, "language": language}
            except Exception as e:
                LOGGER.error("Failed to reconstruct alignment result: %s", e)
                return {}

        # Work with a copy of the result to avoid modifying the original
        cleaned_result = {"text": result.get("text", ""), "segments": []}

        # Copy other potential keys
        for key, value in result.items():
            if key not in ["text", "segments"]:
                cleaned_result[key] = value

        # Get segments from result
        segments = result.get("segments", [])
        if not segments:
            return cleaned_result

        # Determine if this is a CJK language
        is_cjk_lang = is_cjk(language)

        # Clean each segment's text
        for segment in segments:
            # Create a copy of the segment
            cleaned_segment = {}
            for key, value in segment.items():
                cleaned_segment[key] = value

            if "text" in cleaned_segment:
                segment_text = cleaned_segment["text"]
                # Clean the segment text using our helper function
                cleaned_text = clean_alignment_text(segment_text, original_text, is_cjk_lang)
                cleaned_segment["text"] = cleaned_text

            cleaned_result["segments"].append(cleaned_segment)

        # Update the full text in the result
        # Recalculate the full text from cleaned segments
        sep = "" if is_cjk_lang else " "
        full_text = sep.join(segment["text"] for segment in cleaned_result["segments"] if "text" in segment)
        
        # Additional cleanup for full text
        if is_cjk_lang:
            full_text = remove_cjk_spaces(full_text)
        cleaned_result["text"] = full_text

        return cleaned_result
