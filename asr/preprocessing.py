"""
Audio preprocessing module for ASR optimization.

Performs high-pass filtering (room rumble removal), silence trimming,
light noise reduction, and peak normalization on audio streams prior to transcription.
"""

import os
from pathlib import Path
from typing import Optional
from config.config import logger

SAMPLE_RATE = 16000


def convert_audio_to_clean_wav(input_path: str) -> str:
    """
    Convert any input audio container/format (MP3 with ID3 tags, Chrome WebM,
    M4A, OGG, Opus, FLAC) into a pristine 16kHz mono PCM WAV file using FFmpeg.

    This completely prevents libsndfile/mpg123 junk header parsing errors and
    audioread fallbacks.

    Args:
        input_path (str): Path to source audio file.

    Returns:
        str: Path to the clean 16kHz WAV file, or input_path if conversion fails.
    """
    if not os.path.isfile(input_path):
        logger.error(f"Input audio file not found: {input_path}")
        return input_path

    try:
        import ffmpeg
        import soundfile as sf
        import numpy as np

        output_dir = os.path.dirname(input_path)
        clean_wav_path = os.path.join(output_dir, f"clean_{Path(input_path).stem}.wav")

        # If already a valid clean WAV created earlier, return it
        if os.path.isfile(clean_wav_path) and os.path.getsize(clean_wav_path) > 44:
            return clean_wav_path

        out, _ = (
            ffmpeg.input(input_path)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=SAMPLE_RATE)
            .run(capture_stdout=True, capture_stderr=True)
        )
        audio = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0

        if len(audio) == 0:
            logger.warning("FFmpeg decoded empty audio stream; returning original path.")
            return input_path

        sf.write(clean_wav_path, audio, SAMPLE_RATE)
        return clean_wav_path

    except Exception as e:
        logger.warning(f"Fast FFmpeg WAV conversion failed ({e}); returning original path.")
        return input_path


def preprocess_audio(input_path: str) -> Optional[str]:
    """
    Preprocess an audio file to improve ASR transcription accuracy.

    Applies high-pass filtering (70 Hz), silence trimming, light noise reduction,
    and amplitude normalization.

    Args:
        input_path (str): Path to the source audio file.

    Returns:
        Optional[str]: Path to the processed WAV file, or original input_path if processing fails.
    """
    if not os.path.isfile(input_path):
        logger.error(f"Input audio file not found: {input_path}")
        return None

    try:
        import ffmpeg
        import numpy as np
        import librosa
        import soundfile as sf
        from scipy import signal

        logger.info(f"Preprocessing audio: {input_path}")

        output_dir = os.path.dirname(input_path)
        processed_path = os.path.join(output_dir, f"processed_{Path(input_path).stem}.wav")

        # Decode audio to 16kHz mono PCM using ffmpeg
        out, _ = (
            ffmpeg.input(input_path)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=SAMPLE_RATE)
            .run(capture_stdout=True, capture_stderr=True)
        )
        audio = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0

        if len(audio) == 0:
            logger.warning("FFmpeg returned empty audio stream; using original audio.")
            return input_path

        raw_audio = audio.copy()

        try:
            # 1. High-pass filter (cutoff 70 Hz to remove microphone/room rumble)
            if len(audio) > 50:
                nyquist = SAMPLE_RATE / 2
                b, a = signal.butter(4, 70 / nyquist, btype="high")
                audio = signal.filtfilt(b, a, audio)

            # 2. Trim silence
            if len(audio) > 2048:
                audio_trimmed, _ = librosa.effects.trim(
                    audio, top_db=35, frame_length=2048, hop_length=512
                )
                if len(audio_trimmed) > 0:
                    audio = audio_trimmed

            # 3. Light spectral noise reduction (stationary mode for fast execution)
            if len(audio) >= 2048:
                try:
                    import noisereduce as nr
                    audio = nr.reduce_noise(
                        y=audio,
                        sr=SAMPLE_RATE,
                        prop_decrease=0.2,
                        n_fft=1024,
                        win_length=1024,
                        hop_length=512,
                        n_jobs=1,
                        stationary=True,
                    )
                except Exception as nr_err:
                    logger.debug(f"Noise reduction skipped: {nr_err}")

            # 4. Safe amplitude normalization
            if len(audio) > 0 and np.isfinite(audio).all():
                max_val = np.max(np.abs(audio))
                if max_val > 1e-6:
                    audio = audio / max_val
            else:
                audio = raw_audio

        except Exception as filter_err:
            logger.warning(f"DSP filter step failed: {filter_err}. Falling back to raw converted audio.")
            audio = raw_audio

        # Write processed WAV
        sf.write(processed_path, audio, SAMPLE_RATE)
        return processed_path

    except ImportError as imp_err:
        logger.warning(f"Audio DSP dependencies unavailable ({imp_err}). Using fast WAV conversion.")
        return convert_audio_to_clean_wav(input_path)
    except Exception as e:
        logger.error(f"Error preprocessing audio: {e}. Using fast WAV conversion.")
        return convert_audio_to_clean_wav(input_path)