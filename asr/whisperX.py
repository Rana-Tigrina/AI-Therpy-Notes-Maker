import whisperx
import torch
import time
import warnings
import logging
from pathlib import Path
from dotenv import load_dotenv
from .preprocessing import Preprocessor
import os
import gc
import re
from numba.core.errors import NumbaWarning

# Use logger from your config
from config import logger

# Suppress Numba warnings
logging.getLogger('numba').setLevel(logging.WARNING)
warnings.filterwarnings('ignore', module='numba')
os.environ['NUMBA_DISABLE_JIT'] = '0'
load_dotenv()

use_auth_token = os.getenv("HUGGINGFACE_TOKEN")

class WhisperXTranscriber:
    """
    Transcriber using WhisperX for automatic speech recognition with speaker diarization.
    """

    def __init__(self, model_name: str = "base"):
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if self.device == "cuda" else "int8"
            
            self.asr_options = {
                "beam_size": 7,
                "patience": 1.5,
                "compression_ratio_threshold": 2.4,
                "log_prob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "condition_on_previous_text": True,
                "without_timestamps": False,
                "suppress_blank": True,
                "suppress_tokens": [-1],
            }

            self.vad_params = {
                "chunk_size": 20,
                "vad_onset": 0.450,
                "vad_offset": 0.363
            }
            
            # Load model
            base_dir = Path(__file__).resolve().parent.parent
            models_dir = base_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)

            self.model = whisperx.load_model(
                model_name,
                device=self.device,
                compute_type=compute_type,
                download_root=str(models_dir),
                asr_options=self.asr_options,
            )
            
            self.preprocessor = Preprocessor()
            logger.info("WhisperX model loaded successfully")

        except Exception as e:
            logger.error(f"Error initializing WhisperX model: {e}")
            self.model = None
            self.preprocessor = None

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio with speaker diarization.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            str: Transcribed text with speaker labels
        """
        if not self._check_valid(audio_path):
            return ""

        try:
            # Preprocess audio
            logger.info("Processing audio...")
            processed_audio = self.preprocessor.process(audio_path)
            if processed_audio is None:
                logger.error("Audio preprocessing failed")
                return ""

            # Initial transcription
            logger.info("Starting transcription...")
            transcription_result = self._transcribe_audio(processed_audio)
            if not transcription_result:
                logger.error("Initial transcription failed")
                return ""
            
            text = " ".join([segment.get("text", "") for segment in transcription_result.get("segments", [])])
            logger.info(f"Transcription result: {text}")

            # Handle diarization
            logger.info("Starting diarization...")
            diarized_result = self._get_diarized_transcript(processed_audio, transcription_result)
            
            # Format final output
            final_text = self._format_final_output(diarized_result)
            logger.info(f"Final text: {final_text}")
            
            return final_text

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""
        finally:
            self._free_resources()

    def _check_valid(self, audio_path: str) -> bool:
        """Check if model is loaded and file exists."""
        if self.model is None:
            logger.error("WhisperX model is not loaded")
            return False

        if not os.path.isfile(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return False

        return True

    def _transcribe_audio(self, audio_path: str) -> dict:
        """Run transcription on audio file."""
        try:
            start = time.time()
            result = self.model.transcribe(
                audio_path, 
                batch_size=24,
                task='transcribe',
                print_progress=True
            )
            logger.info(f"Transcription completed in {time.time() - start:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {}

    def _free_resources(self):
        """Free GPU memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def post_process_text(self, text: str) -> str:
        """Clean up transcription text."""
        # Remove repeated words
        text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
        
        # Remove repeated phrases
        words = text.split()
        result = []
        i = 0
        while i < len(words):
            result.append(words[i])
            if i + 1 < len(words) and words[i] == words[i + 1]:
                while i + 1 < len(words) and words[i] == words[i + 1]:
                    i += 1
            i += 1
        text = ' '.join(result)

        # Fix spacing and punctuation
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s([?.!"](?:\s|$))', r'\1', text)
        
        return text.strip()