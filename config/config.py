"""
Configuration module for AI Therapy Notes Maker.

Handles environment loading, dynamic storage path resolution (supporting
both local execution and serverless environments such as Vercel), logging setup,
and central application settings.
"""

import os
import sys
import logging
import tempfile
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Suppress known harmless upstream deprecation and third-party notices
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=UserWarning, module="soundfile")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="crisperwhisper")

# Set Hugging Face transformers logging level in environment
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Load environment variables once from root directory
load_dotenv()

# Detect serverless environment (Vercel sets VERCEL=1 or AWS_LAMBDA_FUNCTION_NAME)
IS_SERVERLESS = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))

# Base directory for runtime file storage (uploads, downloads, transcripts)
# In serverless environments, root is read-only so default to system temp directory.
DEFAULT_STORAGE_ROOT = tempfile.gettempdir() if IS_SERVERLESS else "."
STORAGE_ROOT = Path(os.getenv("STORAGE_DIR", DEFAULT_STORAGE_ROOT)).resolve()

UPLOAD_FOLDER = Path(os.getenv("UPLOAD_FOLDER", str(STORAGE_ROOT / "uploads")))
DOWNLOAD_FOLDER = Path(os.getenv("DOWNLOAD_FOLDER", str(STORAGE_ROOT / "downloads")))
TRANSCRIPT_FOLDER = Path(os.getenv("TRANSCRIPT_FOLDER", str(STORAGE_ROOT / "transcript")))

# Ensure runtime directories exist
for folder in (UPLOAD_FOLDER, DOWNLOAD_FOLDER, TRANSCRIPT_FOLDER):
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Fallback to system temp if permissions fail
        fallback = Path(tempfile.gettempdir()) / folder.name
        fallback.mkdir(parents=True, exist_ok=True)

# Allowed audio file extensions for uploads and recordings
ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a", "flac", "webm", "ogg", "mp4", "opus"}

# Maximum upload payload size (default: 50 MB)
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 50 * 1024 * 1024))

# Model & Backend configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
NOTE_GENERATION_MODEL = os.getenv("NOTE_GENERATION_MODEL", os.getenv("GEMINI_MODEL", "gemma-4-31b-it"))
GEMINI_MODEL = NOTE_GENERATION_MODEL  # Backward compatibility alias
GEMINI_ASR_MODEL = os.getenv("GEMINI_ASR_MODEL", "gemini-3.1-flash-lite")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gemini-2.5-flash")
ASR_BACKEND = os.getenv("ASR_BACKEND", "auto").lower()  # 'auto', 'crisper_whisper', or 'gemini'
CRISPER_WHISPER_MODEL = os.getenv("CRISPER_WHISPER_MODEL", "small")

# Demo & Public Rate Limits Configuration
ENABLE_DEMO_LIMITS = os.getenv("ENABLE_DEMO_LIMITS", "true").lower() in ("true", "1", "yes")
MAX_AUDIO_DURATION_SEC = int(os.getenv("MAX_AUDIO_DURATION_SEC", 60))
MAX_USER_UPLOADS = int(os.getenv("MAX_USER_UPLOADS", 2))

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "app.log" if not IS_SERVERLESS else "")

# Configure handlers safely for both serverless and container/local environments
log_handlers = [logging.StreamHandler(sys.stdout)]

if LOG_FILE:
    try:
        log_handlers.append(logging.FileHandler(LOG_FILE, encoding="utf-8"))
    except (OSError, PermissionError):
        pass  # In read-only filesystems, fallback to stream-only logging

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=log_handlers,
)

# Suppress noisy third-party debug loggers
for noisy_logger in ("numba", "urllib3", "httpcore", "httpx", "filelock", "fsspec", "google_genai", "transformers"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

logger = logging.getLogger("therapy_notes")
