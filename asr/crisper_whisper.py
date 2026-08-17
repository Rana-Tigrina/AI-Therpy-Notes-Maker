"""
Speech-to-Text transcription module utilizing CrisperWhisper.

Provides verbatim speech recognition preserving fillers, stutters, repetitions,
and clinical vocal cues without diarization overhead.
"""

import os
import gc
import time
from typing import Optional
from config.config import logger, CRISPER_WHISPER_MODEL
from .preprocessing import preprocess_audio, convert_audio_to_clean_wav


class CrisperWhisperTranscriber:
    """
    Automatic Speech Recognition transcriber using CrisperWhisper
    for high-speed, verbatim transcription.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        backend: str = "auto",
        draft_model: Optional[str] = None,
        use_preprocessing: bool = True,
    ):
        """
        Initialize the CrisperWhisper transcriber.

        Args:
            model_name (str, optional): Model size/id ('turbo', 'large', 'medium', 'small', or HuggingFace ID).
            device (str, optional): Inference device ('cuda' or 'cpu'). Auto-detected if None.
            compute_type (str, optional): Precision ('float16', 'float32', 'int8'). Auto-configured if None.
            backend (str): Backend engine ('auto', 'transformers', 'ct2').
            draft_model (str, optional): Shorthand or HuggingFace ID for speculative decoding.
            use_preprocessing (bool): Whether to apply DSP audio preprocessing before transcription. Defaults to True.
        """
        self.model_name = model_name or CRISPER_WHISPER_MODEL
        self.backend = backend
        self.draft_model = draft_model
        self.use_preprocessing = use_preprocessing
        self.model = None

        try:
            import torch
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            if compute_type is None:
                self.compute_type = "float16" if self.device == "cuda" else "float32"
            else:
                self.compute_type = compute_type

            logger.info(
                f"Loading CrisperWhisper model '{self.model_name}' on {self.device} "
                f"({self.compute_type}, backend={self.backend})..."
            )

            # Silence verbose third-party load report & generation flags warnings
            try:
                import transformers
                transformers.logging.set_verbosity_error()
            except ImportError:
                pass

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

            # Fine-tune Transformers generation config to eliminate max_length conflicts and tokenizer clean-up notices
            if hasattr(self.model, "engine") and hasattr(self.model.engine, "model"):
                if hasattr(self.model.engine.model, "generation_config") and self.model.engine.model.generation_config is not None:
                    self.model.engine.model.generation_config.max_length = None
                if hasattr(self.model.engine, "tokenizer") and self.model.engine.tokenizer is not None:
                    self.model.engine.tokenizer.clean_up_tokenization_spaces = False

            logger.info("CrisperWhisper model initialized successfully.")

        except ImportError as imp_err:
            logger.warning(
                f"CrisperWhisper / PyTorch dependencies not found ({imp_err}). "
                "Local ASR will not be available. Please use Gemini ASR for cloud environments."
            )
        except Exception as e:
            logger.error(f"Failed to initialize CrisperWhisper model: {e}")

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
            str: Transcribed verbatim text, or empty string on failure.
        """
        if self.model is None:
            logger.error("CrisperWhisper model is not available.")
            return ""

        if not os.path.isfile(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return ""

        try:
            target_audio = audio_path
            if self.use_preprocessing:
                processed = preprocess_audio(audio_path)
                if processed:
                    target_audio = processed
            else:
                target_audio = convert_audio_to_clean_wav(audio_path)

            logger.info(f"Running CrisperWhisper transcription (mode={mode}, lang={language})...")
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

            logger.info(f"Transcription completed in {elapsed:.2f}s ({len(transcribed_text.split())} words).")
            return transcribed_text

        except Exception as e:
            logger.error(f"CrisperWhisper transcription error: {e}")
            return ""
        finally:
            self._free_gpu_memory()

    def _free_gpu_memory(self):
        """Free GPU memory cache after inference."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

