"""
Automated Unit and Integration Test Suite for TherapyNotes AI.
Compatible with both pytest and python -m unittest.
Executed automatically in CI/CD pipeline via GitHub Actions.
"""

import io
import unittest
from unittest.mock import MagicMock, patch
from app import app
import routes
from summarizer.summarizer import generate_therapy_notes, validate_therapy_notes
from google.genai.errors import APIError


class TestTherapyNotesApp(unittest.TestCase):
    """Test suite for web routes, quotas, duration checks, and fallback mechanisms."""

    def setUp(self):
        app.config["TESTING"] = True
        app.config["SECRET_KEY"] = "test-secret-key"
        self.client = app.test_client()

    def test_home_page_renders(self):
        """Verify that the landing page renders successfully with 200 OK."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"THERAPYNOTES", response.data)
        self.assertIn(b"Quick Demo Session Available", response.data)

    def test_quota_status_endpoint(self):
        """Verify that quota endpoint returns initial limits."""
        response = self.client.get("/api/quota_status")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["limits_enabled"])
        self.assertEqual(data["max_duration_sec"], 60)
        self.assertEqual(data["max_uploads"], 2)
        self.assertEqual(data["uploads_remaining"], 2)

    def test_process_demo_endpoint(self):
        """Verify that /process_demo returns structured notes without consuming user quota."""
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = "Client discussed anxiety and grounding techniques."
        
        mock_notes = {
            "session_overview": {"summary": "Demo summary"},
            "client_concerns": ["Work stress"],
            "goals_and_progress": {"goals": "Reduce panic attacks"},
            "therapeutic_interventions": ["5-4-3-2-1 technique"],
            "clients_response": {"engagement": "Receptive"},
            "challenges": ["Overthinking"],
            "homework_plan": {"tasks": "Practice box breathing"},
            "next_session_prompts": ["Review anxiety log"]
        }

        with patch("routes._get_active_transcriber", return_value=mock_transcriber), \
             patch("routes.process_with_gemini", return_value=mock_notes):
            response = self.client.post("/process_demo")
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertEqual(data["status"], "success")
            self.assertTrue(data["is_demo"])
            self.assertIn("validated_therapy_notes", data)
            self.assertEqual(data["uploads_remaining"], 2)  # Quota unchanged

    def test_duration_limit_enforcement(self):
        """Verify that audio longer than 60s is rejected with 400 Bad Request."""
        audio_stream = io.BytesIO(b"RIFFdummydata")
        response = self.client.post(
            "/process_audio_file",
            data={"audio": (audio_stream, "test.wav"), "duration": "95"},
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("exceeds the 60s limit", data["error"])

    def test_custom_upload_quota_limit(self):
        """Verify that the 2-upload quota is strictly enforced with 429 Too Many Requests."""
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = "Sample consultation transcript."

        with patch("routes._get_active_transcriber", return_value=mock_transcriber), \
             patch("routes.process_with_gemini", return_value={"session_overview": "test"}):

            # Upload 1
            r1 = self.client.post(
                "/process_audio_file",
                data={"audio": (io.BytesIO(b"audio1"), "test1.wav"), "duration": "15"},
                content_type="multipart/form-data"
            )
            self.assertEqual(r1.status_code, 200)
            self.assertEqual(r1.get_json()["uploads_remaining"], 1)

            # Upload 2
            r2 = self.client.post(
                "/process_audio_file",
                data={"audio": (io.BytesIO(b"audio2"), "test2.wav"), "duration": "20"},
                content_type="multipart/form-data"
            )
            self.assertEqual(r2.status_code, 200)
            self.assertEqual(r2.get_json()["uploads_remaining"], 0)

            # Upload 3 (Exceeded)
            r3 = self.client.post(
                "/process_audio_file",
                data={"audio": (io.BytesIO(b"audio3"), "test3.wav"), "duration": "10"},
                content_type="multipart/form-data"
            )
            self.assertEqual(r3.status_code, 429)
            self.assertTrue(r3.get_json()["quota_exhausted"])

    def test_fallback_mechanism_on_503_error(self):
        """Verify that note generation automatically fails over to FALLBACK_MODEL on 503 errors."""
        call_counts = {"primary": 0, "fallback": 0}

        def mock_generate_content(model, contents, config=None):
            if model == "gemma-4-31b-it":
                call_counts["primary"] += 1
                # Simulate 503 UNAVAILABLE spike
                raise APIError(503, "This model is currently experiencing high demand.", None)
            else:
                call_counts["fallback"] += 1
                mock_resp = MagicMock()
                mock_resp.text = '{"session_overview": "Fallback success"}'
                return mock_resp

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = mock_generate_content

        with patch("summarizer.summarizer.client", mock_client):
            result = generate_therapy_notes("Patient reported anxiety symptoms.")
            self.assertEqual(result, {"session_overview": "Fallback success"})
            self.assertEqual(call_counts["primary"], 1)
            self.assertEqual(call_counts["fallback"], 1)


if __name__ == "__main__":
    unittest.main()
