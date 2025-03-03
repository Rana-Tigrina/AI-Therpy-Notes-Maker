# AI Therapy Notes Maker

## Overview

 AI  Therapy Notes Maker is a comprehensive system designed to transcribe, process, and generate detailed therapy notes from audio recordings. Leveraging advanced Automatic Speech Recognition (ASR) and natural language processing, the pipeline ensures accurate and insightful summaries, optimized for therapeutic applications.

## Features

- **Automatic Transcription**: Converts audio files into text using WhisperXTranscriber.
- **Audio Preprocessing**: Optimizes audio for ASR accuracy.
- **Therapy Notes Generation**: Produces structured therapy notes with identified techniques and session prompts.
- **API Integration**: Provides RESTful endpoints for seamless integration with other applications.
- **Document Generation**: Creates well-formatted therapy notes documents in DOCX format.

## Architecture

![Architecture Diagram](docs/architecture.png)

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
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Configure Environment Variables**

    Create a `.env` file in the [config](http://_vscodecontentref_/0) directory with the following variables:
    ```env
    UPLOAD_FOLDER=./uploads
    LOG_FILE=app.log
    GEMINI_API_KEY=your_gemini_api_key
    ```

5. **Set Up Folders**
    ```bash
    mkdir -p uploads downloads transcript
    ```

## Usage

### Running the API Server

```bash
python app.py

The server will start on http://localhost:5000.
```