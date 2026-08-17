"""
Speech recognition (ASR) subpackage.

Provides transcriber implementations for both local execution (CrisperWhisper)
and serverless/cloud execution (Gemini Multimodal Audio).
"""

from typing import Union
from config.config import ASR_BACKEND, IS_SERVERLESS, logger
from .crisper_whisper import CrisperWhisperTranscriber
from .gemini_asr import GeminiAudioTranscriber
from .preprocessing import preprocess_audio, convert_audio_to_clean_wav


def get_transcriber() -> Union[CrisperWhisperTranscriber, GeminiAudioTranscriber]:
    """
    Factory function to instantiate the appropriate ASR transcriber based on
    environment configuration (ASR_BACKEND, VERCEL/serverless detection).

    Returns:
        CrisperWhisperTranscriber or GeminiAudioTranscriber instance.
    """
    if IS_SERVERLESS or ASR_BACKEND == "gemini":
        logger.info("Initializing Gemini Cloud Audio Transcriber (serverless/cloud mode).")
        return GeminiAudioTranscriber()

    if ASR_BACKEND == "crisper_whisper":
        logger.info("Initializing CrisperWhisper local transcriber.")
        return CrisperWhisperTranscriber()

    # 'auto' mode: try local CrisperWhisper first; fallback to Gemini if model is unavailable
    try:
        local_asr = CrisperWhisperTranscriber()
        if local_asr.model is not None:
            return local_asr
        logger.warning("Local CrisperWhisper unavailable; falling back to Gemini Cloud ASR.")
        return GeminiAudioTranscriber()
    except Exception as e:
        logger.warning(f"Error checking local ASR ({e}); falling back to Gemini Cloud ASR.")
        return GeminiAudioTranscriber()


__all__ = [
    "CrisperWhisperTranscriber",
    "GeminiAudioTranscriber",
    "get_transcriber",
    "preprocess_audio",
    "convert_audio_to_clean_wav",
]