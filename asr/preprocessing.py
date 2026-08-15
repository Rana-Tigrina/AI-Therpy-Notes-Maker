import os
import ffmpeg
import numpy as np
import librosa
import noisereduce as nr
from pathlib import Path
import soundfile as sf
from scipy import signal

from config.config import logger

SAMPLE_RATE = 16000

def preprocess_audio(input_path: str) -> str:
    """
    Process audio optimized for ASR accuracy in therapy sessions.
    """
    try:
        logger.info(f"Preprocessing audio: {input_path}")
        
        # Create output directory in same location as input file
        output_dir = os.path.dirname(input_path)
        processed_path = os.path.join(output_dir, f"processed_{Path(input_path).stem}.wav")

        # Load audio file via ffmpeg
        out, _ = (
            ffmpeg.input(input_path)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=SAMPLE_RATE)
            .run(capture_stdout=True, capture_stderr=True)
        )
        audio = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0

        if len(audio) == 0:
            logger.error("FFmpeg returned empty audio stream.")
            return input_path

        raw_audio = audio.copy()

        try:
            # 1. Remove low-frequency noise (if audio has sufficient length)
            if len(audio) > 50:
                nyquist = SAMPLE_RATE / 2
                cutoff = 70  # Hz - removes room rumble while preserving speech
                b, a = signal.butter(4, cutoff / nyquist, btype='high')
                audio = signal.filtfilt(b, a, audio)

            # 2. Trim silence (only if result is non-empty)
            if len(audio) > 2048:
                audio_trimmed, _ = librosa.effects.trim(
                    audio, top_db=35, frame_length=2048, hop_length=512
                )
                if len(audio_trimmed) > 0:
                    audio = audio_trimmed

            # 3. Noise reduction (if audio is long enough for n_fft)
            if len(audio) >= 2048:
                audio = nr.reduce_noise(
                    y=audio,
                    sr=SAMPLE_RATE,
                    prop_decrease=0.3,
                    n_fft=2048,
                    win_length=2048,
                    hop_length=512,
                    n_jobs=1,
                    stationary=False
                )

            # 4. Safe normalization
            if len(audio) > 0 and np.isfinite(audio).all():
                max_val = np.max(np.abs(audio))
                if max_val > 1e-6:
                    audio = audio / max_val
            else:
                audio = raw_audio

        except Exception as filter_err:
            logger.warning(f"DSP filter step failed: {filter_err}. Falling back to raw converted audio.")
            audio = raw_audio

        # Save the processed audio
        sf.write(processed_path, audio, SAMPLE_RATE)
        return processed_path
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}. Using original audio file.")
        return input_path

class Preprocessor:
    def process(self, audio_path: str) -> str:
        """
        Process audio file for CrisperWhisper transcription.
        
        Args:
            audio_path (str): Path to the input audio file.
            
        Returns:
            str: Path to the processed audio file or None if processing fails.
        """
        try:
            # Verify input audio file exists
            if not os.path.isfile(audio_path):
                logger.error(f"Input audio file does not exist: {audio_path}")
                return None

            # Process the audio
            return preprocess_audio(audio_path)
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            return None