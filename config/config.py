import os
import logging

from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (root first, then config/)
load_dotenv()
config_env = Path(__file__).resolve().parent / '.env'
if config_env.exists():
    load_dotenv(dotenv_path=config_env)

# Directory where uploaded files are stored
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', './uploads')

# Allowed file extensions for audio uploads & live recordings
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'webm', 'ogg', 'mp4', 'opus'}

# Log file configuration
LOG_FILE = os.getenv('LOG_FILE', 'app.log')

# Configure logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# Silence noisy third-party debug loggers
for noisy_logger in ('numba', 'urllib3', 'httpcore', 'httpx', 'filelock', 'fsspec'):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)