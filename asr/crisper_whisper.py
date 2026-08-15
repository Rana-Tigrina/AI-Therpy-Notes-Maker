import os
import gc
import time
import torch
import logging
from pathlib import Path
from dotenv import load_dotenv

from config.config import logger
from .preprocessing import Preprocessor

load_dotenv()


class CrisperWhisperTranscriber:
    """
    Automatic Speech Recognition transcriber using CrisperWhisper 2.0 Turbo
    for high-speed, verbatim transcription (preserving fillers, stutters, repetitions, and vocal events).
    """

    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        compute_type: str = None,
        backend: str = "auto",
        draft_model: str = None,
        use_preprocessing: bool = False,
    ):
        """
        Initialize the CrisperWhisper transcriber.

        Args:
            model_name (str): Model size/id ('turbo', 'large', 'medium', 'small', or HuggingFace ID).
                              Defaults to 'turbo' (nyralabs/CrisperWhisper2.0_turbo).
            device (str): Inference device ('cuda' or 'cpu'). Auto-detected if None.
            compute_type (str): Precision type ('float16', 'float32', 'int8'). Auto-configured if None.
            backend (str): Backend engine ('auto', 'transformers', 'ct2').
            draft_model (str): Optional draft model shorthand or ID for speculative decoding.
            use_preprocessing (bool): Whether to run DSP audio preprocessing filter before transcription.
        """
        try:
            self.model_name = model_name or os.getenv("CRISPER_WHISPER_MODEL", "small")
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

            if compute_type is None:
                self.compute_type = "float16" if self.device == "cuda" else "float32"
            else:
                self.compute_type = compute_type

            self.backend = backend
            self.draft_model = draft_model
            self.use_preprocessing = use_preprocessing
            self.preprocessor = Preprocessor() if use_preprocessing else None

            logger.info(
                f"Initializing CrisperWhisperModel (model={self.model_name}, device={self.device}, "
                f"compute_type={self.compute_type}, backend={self.backend})"
            )

            from crisperwhisper import CrisperWhisperModel

            init_kwargs = {
                "model_name_or_path": self.model_name,
                "backend": self.backend,
                "compute_type": self.compute_type,
                "device": self.device,
            }
            if self.draft_model:
                init_kwargs["draft_model"] = self.draft_model

            self.model = CrisperWhisperModel(**init_kwargs)
            logger.info("CrisperWhisper model loaded successfully.")

        except Exception as e:
            logger.error(f"Error initializing CrisperWhisper model: {e}")
            self.model = None

    def transcribe(
        self,
        audio_path: str,
        language: str = "en",
        mode: str = "verbatim",
        word_timestamps: bool = False,
        longform_strategy: str = "continuation",
        hallucination_mitigation: bool = True,
        **kwargs,
    ) -> str:
        """
        Transcribe audio into text using CrisperWhisper in verbatim mode.

        Args:
            audio_path (str): Path to audio file.
            language (str): Language code (default: 'en').
            mode (str): Transcription mode ('verbatim' or 'intended'). Defaults to 'verbatim'.
            word_timestamps (bool): Whether to calculate word-level timestamps.
            longform_strategy (str): Longform chunking strategy ('continuation', 'chunked_lcs', 'token_lcs').
            hallucination_mitigation (bool): Enable repetition detection and repair.

        Returns:
            str: Transcribed verbatim text.
        """
        if not self._check_valid(audio_path):
            return ""

        try:
            target_audio = audio_path
            if self.use_preprocessing and self.preprocessor:
                logger.info("Running audio preprocessing...")
                processed = self.preprocessor.process(audio_path)
                if processed:
                    target_audio = processed

            logger.info(f"Starting CrisperWhisper transcription (mode={mode}, language={language})...")
            start_time = time.time()

            result = self.model.transcribe(
                target_audio,
                language=language,
                mode=mode,
                word_timestamps=word_timestamps,
                longform_strategy=longform_strategy,
                hallucination_mitigation=hallucination_mitigation,
                **kwargs,
            )

            elapsed = time.time() - start_time
            transcribed_text = result.text.strip() if hasattr(result, "text") else str(result).strip()

            logger.info(f"CrisperWhisper transcription completed in {elapsed:.2f}s")
            logger.debug(f"Verbatim Transcript: {transcribed_text}")

            return transcribed_text

        except Exception as e:
            logger.error(f"CrisperWhisper transcription failed: {e}")
            return ""
        finally:
            self._free_resources()

    def _check_valid(self, audio_path: str) -> bool:
        """Check if model is initialized and audio file exists."""
        if self.model is None:
            logger.error("CrisperWhisper model is not loaded.")
            return False

        if not os.path.isfile(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return False

        return True

    def _free_resources(self):
        """Free GPU memory and run garbage collection."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
