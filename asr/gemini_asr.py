"""
Cloud-native Audio Speech Recognition (ASR) using Google Gemini.

Provides fast, high-accuracy verbatim audio transcription via Gemini Multimodal Audio,
requiring zero local PyTorch/FFmpeg dependencies—ideal for serverless deployments (Vercel).
"""

import os
from pathlib import Path
from google import genai
from google.genai import types
from config.config import logger, GEMINI_API_KEY, GEMINI_ASR_MODEL, FALLBACK_MODEL


class GeminiAudioTranscriber:
    """
    Cloud-based audio transcriber utilizing Gemini multimodal capabilities.
    Accepts audio files (mp3, wav, m4a, webm, ogg, etc.) and produces verbatim transcripts.
    Includes automated fallback to fallback model if primary model is unavailable.
    """

    def __init__(self, api_key: str = None, model_name: str = None):
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name or GEMINI_ASR_MODEL
        if not self.api_key:
            logger.warning("GEMINI_API_KEY is not set. Gemini audio transcription will fail unless configured.")
            self.client = None
        else:
            self.client = genai.Client(api_key=self.api_key)

    def transcribe(self, audio_path: str, **kwargs) -> str:
        """
        Transcribe an audio file using Google Gemini Multimodal Audio.
        Tries primary model (gemini-3.1-flash-lite) with automated fallback to gemini-2.5-flash.

        Args:
            audio_path (str): Absolute or relative path to the audio file.

        Returns:
            str: Verbatim transcription text, or empty string on failure.
        """
        if not self.client:
            logger.error("Gemini client is not initialized. Please configure GEMINI_API_KEY.")
            return ""

        path_obj = Path(audio_path)
        if not path_obj.is_file():
            logger.error(f"Audio file does not exist: {audio_path}")
            return ""

        # Map extension to audio mime type
        ext = path_obj.suffix.lower().lstrip(".")
        mime_map = {
            "wav": "audio/wav",
            "mp3": "audio/mp3",
            "m4a": "audio/m4a",
            "aac": "audio/aac",
            "webm": "audio/webm",
            "ogg": "audio/ogg",
            "opus": "audio/ogg",
            "flac": "audio/flac",
            "mp4": "audio/mp4",
        }
        mime_type = mime_map.get(ext, "audio/mp3")

        models_to_try = [self.model_name]
        if FALLBACK_MODEL and FALLBACK_MODEL not in models_to_try:
            models_to_try.append(FALLBACK_MODEL)

        try:
            logger.info(f"Uploading audio file {path_obj.name} to Gemini API...")
            audio_file = self.client.files.upload(
                file=str(path_obj),
                config=types.UploadFileConfig(mime_type=mime_type)
            )

            prompt = (
                "Please provide a complete, precise, verbatim transcription of this therapy/consultation session audio. "
                "Preserve all speaker turns, pauses, filler words, repetitions, and vocal events accurately. "
                "Output ONLY the transcribed conversation text without meta-commentary or introduction."
            )

            transcript = ""
            for idx, model_name in enumerate(models_to_try):
                try:
                    logger.info(f"Requesting transcription via {model_name}...")
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=[audio_file, prompt]
                    )
                    transcript = response.text.strip() if response.text else ""
                    if transcript:
                        logger.info(f"Gemini transcription successful using {model_name} ({len(transcript.split())} words).")
                        break
                except Exception as gen_err:
                    if idx < len(models_to_try) - 1:
                        next_model = models_to_try[idx + 1]
                        logger.warning(
                            f"Transcription failed with '{model_name}' ({gen_err}); attempting automatic fallback to '{next_model}'."
                        )
                    else:
                        logger.error(f"Gemini audio transcription failed across all candidate models ({models_to_try}): {gen_err}")

            # Cleanup uploaded file from Gemini storage
            try:
                self.client.files.delete(name=audio_file.name)
            except Exception as del_err:
                logger.debug(f"Temporary remote audio cleanup notice: {del_err}")

            return transcript

        except Exception as e:
            logger.error(f"Gemini audio upload/transcription failed: {e}")
            return ""
