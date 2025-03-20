import os
import logging

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Directory where uploaded files are stored
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', './uploads')

# Allowed file extensions for audio uploads
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac'}

# Log file configuration
LOG_FILE = os.getenv('LOG_FILE', 'app.log')

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)