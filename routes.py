"""
REST API Routes Blueprint for audio ingestion, transcription, and therapy notes generation.

Exposes endpoints for processing uploaded or recorded audio sessions, testing with pre-built demo audio,
and downloading generated Word (.docx) clinical reports with session-based demo rate limits.
"""

import os
import uuid
import shutil
from pathlib import Path
from flask import Blueprint, request, jsonify, send_from_directory, session
from werkzeug.utils import secure_filename

from asr import get_transcriber
from summarizer import process_with_gemini, TherapyNotesDocumentGenerator
from config.config import (
    logger,
    ALLOWED_EXTENSIONS,
    UPLOAD_FOLDER,
    DOWNLOAD_FOLDER,
    TRANSCRIPT_FOLDER,
    ENABLE_DEMO_LIMITS,
    MAX_AUDIO_DURATION_SEC,
    MAX_USER_UPLOADS,
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


@api_bp.route("/api/quota_status", methods=["GET"])
def quota_status():
    """
    Get the remaining audio processing credits for the active user session.
    """
    used = session.get("custom_uploads", 0)
    remaining = max(0, MAX_USER_UPLOADS - used) if ENABLE_DEMO_LIMITS else 999
    return jsonify({
        "limits_enabled": ENABLE_DEMO_LIMITS,
        "max_duration_sec": MAX_AUDIO_DURATION_SEC,
        "max_uploads": MAX_USER_UPLOADS,
        "uploads_used": used,
        "uploads_remaining": remaining,
    }), 200


@api_bp.route("/process_demo", methods=["POST"])
def process_demo():
    """
    Process the built-in sample therapy session audio (demo.mp3).
    Does NOT consume the user's custom upload limit, allowing unlimited test trials.
    """
    try:
        demo_path = Path(__file__).resolve().parent / "static" / "demo.mp3"
        if not demo_path.is_file():
            # Fallback to local static path
            demo_path = Path("static/demo.mp3")

        if not demo_path.is_file():
            return jsonify({"error": "Demo audio file is missing on the server."}), 404

        unique_prefix = str(uuid.uuid4())
        unique_audio_name = f"{unique_prefix}_demo.mp3"
        filepath = UPLOAD_FOLDER / unique_audio_name
        shutil.copyfile(str(demo_path), str(filepath))

        # 1. Transcribe audio
        transcriber = _get_active_transcriber()
        logger.info(f"Processing demo session audio: {filepath.name}")
        transcript = transcriber.transcribe(str(filepath))

        if not transcript:
            return jsonify({"error": "Demo transcription failed. Please check service connectivity."}), 500

        # Save transcript
        transcript_filename = f"{unique_prefix}_demo.txt"
        transcript_filepath = TRANSCRIPT_FOLDER / transcript_filename
        try:
            with open(transcript_filepath, "w", encoding="utf-8") as f:
                f.write(transcript)
        except OSError as e:
            logger.warning(f"Could not persist transcript to disk ({e}); continuing with processing.")

        # 2. Synthesize structured clinical therapy notes
        logger.info("Synthesizing clinical notes for demo session...")
        therapy_notes = process_with_gemini(transcript)

        # 3. Generate Word document (.docx) report
        doc_generator = TherapyNotesDocumentGenerator()
        doc_generator.generate_document(therapy_notes)

        docx_filename = f"{unique_prefix}_therapy_notes.docx"
        docx_filepath = DOWNLOAD_FOLDER / docx_filename
        doc_generator.save_document(str(docx_filepath))
        logger.info(f"Generated demo clinical DOCX report at: {docx_filepath.name}")

        used = session.get("custom_uploads", 0)
        remaining = max(0, MAX_USER_UPLOADS - used) if ENABLE_DEMO_LIMITS else 999

        return jsonify({
            "status": "success",
            "is_demo": True,
            "validated_therapy_notes": therapy_notes,
            "docx_url": f"/download/{docx_filename}",
            "uploads_remaining": remaining,
        }), 200

    except Exception as e:
        logger.error(f"Error in process_demo endpoint: {e}", exc_info=True)
        return jsonify({"error": f"Demo processing error: {str(e)}"}), 500


@api_bp.route("/process_audio_file", methods=["POST"])
def process_audio_file():
    """
    Process an audio recording or uploaded session file with duration & quota verification.
    Transcribes audio, synthesizes structured clinical notes via Gemma / Gemini,
    and generates a downloadable Word (.docx) report.

    Returns:
        JSON response with status, validated_therapy_notes, docx_url, and quota info.
    """
    try:
        # Check session upload rate limits if demo limits are enabled
        current_used = session.get("custom_uploads", 0)
        if ENABLE_DEMO_LIMITS and current_used >= MAX_USER_UPLOADS:
            return jsonify({
                "error": f"Demo limit reached: You have used your {MAX_USER_UPLOADS} free custom audio credits. "
                         f"You can still test with the 'Try Demo Session' button or host your own instance with custom API keys.",
                "quota_exhausted": True
            }), 429

        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided in request."}), 400

        file = request.files["audio"]
        if not file.filename:
            return jsonify({"error": "No file selected."}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Unsupported audio format. Allowed: wav, mp3, m4a, webm, ogg, mp4."}), 400

        # Validate duration if passed by client form
        duration_str = request.form.get("duration", "")
        if duration_str:
            try:
                duration_val = float(duration_str)
                if ENABLE_DEMO_LIMITS and duration_val > (MAX_AUDIO_DURATION_SEC + 5):  # 5s grace buffer
                    return jsonify({
                        "error": f"Audio length ({int(duration_val)}s) exceeds the {MAX_AUDIO_DURATION_SEC}s limit for public testing. "
                                 f"Please provide an audio clip of {MAX_AUDIO_DURATION_SEC} seconds or less."
                    }), 400
            except ValueError:
                pass

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

        # 2. Synthesize structured clinical therapy notes via Google Gemini / Gemma
        logger.info("Synthesizing clinical notes from transcript with Google GenAI...")
        therapy_notes = process_with_gemini(transcript)

        # 3. Generate Word document (.docx) report
        doc_generator = TherapyNotesDocumentGenerator()
        doc_generator.generate_document(therapy_notes)

        docx_filename = f"{unique_prefix}_therapy_notes.docx"
        docx_filepath = DOWNLOAD_FOLDER / docx_filename
        doc_generator.save_document(str(docx_filepath))
        logger.info(f"Generated clinical DOCX report at: {docx_filepath.name}")

        # Increment session upload usage
        session["custom_uploads"] = current_used + 1
        remaining = max(0, MAX_USER_UPLOADS - session["custom_uploads"]) if ENABLE_DEMO_LIMITS else 999

        # 4. Return structured payload to client
        return jsonify({
            "status": "success",
            "validated_therapy_notes": therapy_notes,
            "docx_url": f"/download/{docx_filename}",
            "uploads_remaining": remaining,
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
