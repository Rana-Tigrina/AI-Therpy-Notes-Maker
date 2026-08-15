# AI Therapy Notes Maker

## Overview

**AI Therapy Notes Maker** is a clinical documentation system designed to transcribe audio recordings and generate structured therapy notes. Powered by **Crisper-Whisper 2.0 Turbo** (`nyralabs/CrisperWhisper2.0_turbo`) for high-speed **verbatim transcription** and Google Gemini for clinical note synthesis, the pipeline captures every clinical nuance (fillers, stutters, repetitions, and vocal events) to produce accurate summaries in DOCX format.

## Features

- **Verbatim Speech Recognition**: Powered by `Crisper-Whisper 2.0 Turbo` (capturing all fillers, pauses, repetitions, and vocal cues without diarization overhead).
- **Fast Inference**: Lightweight 4-decoder-layer architecture optimized for rapid clinical transcription.
- **Audio Preprocessing**: Optional DSP filter optimization (filtering rumble, normalization).
- **Clinical Therapy Notes Generation**: Produces structured clinical notes (DAP, SOAP, MSE, and thematic summaries) using Google Gemini.
- **DOCX Document Generation**: Automatically formats and exports therapy notes into structured Word documents.
- **RESTful API & Interactive UI**: Clean web interface and REST endpoints for session uploads and real-time processing.

## Installation

### Prerequisites

- Python 3.8+
- Git
- FFmpeg

### Steps

1. **Clone the Repository**
    ```bash
    git clone https://github.com/Rana-Tigrina/AI-Therpy-Notes-Maker.git
    cd AI-Therpy-Notes-Maker
    ```

2. **Create a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Configure Environment Variables**

    Create a `.env` file in the root directory:
    ```env
    UPLOAD_FOLDER=./uploads
    LOG_FILE=app.log
    GEMINI_API_KEY=your_gemini_api_key
    CRISPER_WHISPER_MODEL=turbo
    ```

5. **Set Up Folders**
    ```bash
    mkdir -p uploads downloads transcript
    ```

## Usage

### Running the Application

```bash
python app.py
```

The server will start on `http://localhost:5000`.