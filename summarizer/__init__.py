"""
Summarization subpackage for clinical notes processing and document generation.
"""

from .summarizer import process_with_gemini
from .docgen import TherapyNotesDocumentGenerator

__all__ = ["process_with_gemini", "TherapyNotesDocumentGenerator"]