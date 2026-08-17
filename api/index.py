"""
Vercel Serverless Function Entry Point.

Exposes the Flask WSGI application instance for Vercel Python Serverless Runtime.
"""

import sys
from pathlib import Path

# Ensure project root is on Python path for serverless imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import app  # noqa: E402

# WSGI handler for Vercel
handler = app
