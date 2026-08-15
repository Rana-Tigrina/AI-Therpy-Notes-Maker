import uuid
from pathlib import Path
# pyrefly: ignore [missing-import]
from flask import Blueprint, request, jsonify, send_from_directory
# pyrefly: ignore [missing-import]
from werkzeug.utils import secure_filename
from asr.crisper_whisper import CrisperWhisperTranscriber
from summarizer import process_with_gemini, TherapyNotesDocumentGenerator
from summarizer.summarizer import process_with_local
from config.config import logger, ALLOWED_EXTENSIONS

api_bp = Blueprint('api', __name__)

UPLOAD_FOLDER = Path('./uploads')
DOWNLOAD_FOLDER = Path('./downloads')
TRANSCRIPT_FOLDER = Path('./transcript')

for _dir in (UPLOAD_FOLDER, DOWNLOAD_FOLDER, TRANSCRIPT_FOLDER):
    _dir.mkdir(parents=True, exist_ok=True)

transcriber = CrisperWhisperTranscriber(model_name="small")

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@api_bp.route('/process_audio_file', methods=['POST'])
def process_audio_file():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided.'}), 400

        file = request.files['audio']
        if not file.filename:
            return jsonify({'error': 'No file selected.'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Unsupported file type.'}), 400

        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = UPLOAD_FOLDER / unique_filename
        file.save(str(filepath))

        # 1. Transcribe the audio file
        transcript = transcriber.transcribe(str(filepath))
        if not transcript:
            return jsonify({'error': 'Transcription failed.'}), 500

        # Save transcript to file for record
        transcript_filename = f"{Path(unique_filename).stem}.txt"
        transcript_filepath = TRANSCRIPT_FOLDER / transcript_filename
        with open(transcript_filepath, 'w', encoding='utf-8') as f:
            f.write(transcript)

        # 2. Generate therapy notes using Google Gemini
        logger.info("Generating clinical notes using Google Gemini...")
        therapy_notes = process_with_gemini(transcript)

        # 4. Generate the DOCX document
        doc_generator = TherapyNotesDocumentGenerator()
        doc_generator.generate_document(therapy_notes)
        
        # Save DOCX to DOWNLOAD_FOLDER
        docx_filename = f"{uuid.uuid4()}_therapy_notes.docx"
        docx_filepath = DOWNLOAD_FOLDER / docx_filename
        doc_generator.save_document(str(docx_filepath))
        
        logger.info(f"Successfully generated DOCX notes at {docx_filepath}")

        # 5. Return response format expected by frontend
        return jsonify({
            'status': 'success',
            'validated_therapy_notes': therapy_notes,
            'docx_url': f'/download/{docx_filename}'
        }), 200

    except Exception as e:
        logger.error(f"Error in process_audio_file: {e}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    try:
        return send_from_directory(str(DOWNLOAD_FOLDER.resolve()), filename, as_attachment=True)
    except Exception as e:
        logger.error(f"Error downloading file {filename}: {e}")
        return jsonify({'error': 'File not found or access denied.'}), 404
