"""
REST API Routes Blueprint for audio ingestion, transcription, and therapy notes generation.

Exposes endpoints for processing uploaded or recorded audio sessions and downloading
generated Word (.docx) clinical reports.
"""

import uuid
from pathlib import Path
from flask import Blueprint, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from asr import get_transcriber
from summarizer import process_with_gemini, TherapyNotesDocumentGenerator
from config.config import (
    logger,
    ALLOWED_EXTENSIONS,
    UPLOAD_FOLDER,
    DOWNLOAD_FOLDER,
    TRANSCRIPT_FOLDER,
)

api_bp = Blueprint("api", __name__)

# Transcriber instance (lazily resolved via factory)
_transcriber = None


def _get_active_transcriber():
    """Lazily initialize and cache active ASR transcriber instance."""
    global _transcriber
    if _transcriber is None:
        _transcriber = get_transcriber()
    return _transcriber


def allowed_file(filename: str) -> bool:
    """Validate if file extension is permitted."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@api_bp.route("/process_audio_file", methods=["POST"])
def process_audio_file():
    """
    Process an audio recording or uploaded session file.
    Transcribes audio, synthesizes structured clinical notes via Gemini,
    and generates a downloadable Word (.docx) report.

    Returns:
        JSON response with status, validated_therapy_notes, and docx_url.
    """
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided in request."}), 400

        file = request.files["audio"]
        if not file.filename:
            return jsonify({"error": "No file selected."}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Unsupported audio format. Allowed: wav, mp3, m4a, webm, ogg, mp4."}), 400

        # Save audio with secure unique filename
        filename = secure_filename(file.filename)
        unique_prefix = str(uuid.uuid4())
        unique_audio_name = f"{unique_prefix}_{filename}"
        filepath = UPLOAD_FOLDER / unique_audio_name
        file.save(str(filepath))

        # 1. Transcribe audio (via local CrisperWhisper or cloud Gemini ASR)
        transcriber = _get_active_transcriber()
        logger.info(f"Transcribing audio file: {filepath.name}")
        transcript = transcriber.transcribe(str(filepath))

        if not transcript:
            return jsonify({"error": "Transcription failed or produced empty text. Please check audio quality."}), 500

        # Save transcript text for record
        transcript_filename = f"{unique_prefix}_{Path(filename).stem}.txt"
        transcript_filepath = TRANSCRIPT_FOLDER / transcript_filename
        try:
            with open(transcript_filepath, "w", encoding="utf-8") as f:
                f.write(transcript)
        except OSError as e:
            logger.warning(f"Could not persist transcript to disk ({e}); continuing with processing.")

        # 2. Synthesize structured clinical therapy notes via Google Gemini
        logger.info("Synthesizing clinical notes from transcript with Google Gemini...")
        therapy_notes = process_with_gemini(transcript)

        # 3. Generate Word document (.docx) report
        doc_generator = TherapyNotesDocumentGenerator()
        doc_generator.generate_document(therapy_notes)

        docx_filename = f"{unique_prefix}_therapy_notes.docx"
        docx_filepath = DOWNLOAD_FOLDER / docx_filename
        doc_generator.save_document(str(docx_filepath))
        logger.info(f"Generated clinical DOCX report at: {docx_filepath.name}")

        # 4. Return structured payload to client
        return jsonify({
            "status": "success",
            "validated_therapy_notes": therapy_notes,
            "docx_url": f"/download/{docx_filename}",
        }), 200

    except Exception as e:
        logger.error(f"Error in process_audio_file endpoint: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@api_bp.route("/download/<filename>", methods=["GET"])
def download_file(filename: str):
    """
    Download a generated DOCX clinical report.

    Args:
        filename (str): Name of the document file in downloads folder.

    Returns:
        File attachment or 404 error JSON.
    """
    try:
        sanitized = secure_filename(filename)
        return send_from_directory(
            str(DOWNLOAD_FOLDER.resolve()),
            sanitized,
            as_attachment=True,
            download_name="therapy_notes.docx",
        )
    except Exception as e:
        logger.error(f"Error downloading file '{filename}': {e}")
        return jsonify({"error": "File not found or access denied."}), 404
