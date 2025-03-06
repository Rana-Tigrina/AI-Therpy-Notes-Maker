import logging
import os

#write a simple logger 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', './uploads')

