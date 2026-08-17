"""
Clinical Summarization & Therapy Notes Generation Module.

Analyzes transcribed therapy session transcripts and synthesizes structured,
evidence-based clinical documentation (overview, client concerns, goals, interventions,
response, challenges, homework, next session prompts) using Google Gemini.
"""

import json
import logging
from typing import Any, Dict
from google import genai
from google.genai import types
from config.config import logger, GEMINI_API_KEY, NOTE_GENERATION_MODEL, FALLBACK_MODEL

# Initialize Google GenAI client
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Curated reference list of clinical therapeutic modalities
KNOWN_TECHNIQUES = [
    "Cognitive-Behavioral Therapy (CBT)",
    "Solution-Focused Brief Therapy (SFBT)",
    "Behavioral Activation",
    "Person-Centered / Humanistic Therapy",
    "Stress Management & Grounding Techniques",
    "Narrative Therapy",
    "Psychoeducation",
    "Strength-Based Approach",
    "Socratic Questioning",
    "Motivational Interviewing (MI)",
    "Mindfulness-Based Interventions (MBCT/MBSR)",
    "Emotion-Focused Therapy (EFT)",
    "Open-Ended & Reflective Questioning",
    "Values Clarification",
    "Problem-Solving Therapy",
    "Acceptance and Commitment Therapy (ACT)",
]


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Extract and parse JSON object from a model response string.

    Args:
        text (str): Raw string output from the model.

    Returns:
        Dict[str, Any]: Parsed JSON dictionary, or empty dict on failure.
    """
    if not text:
        return {}

    text = text.strip()
    # Find JSON boundaries if surrounded by markdown code blocks or extra text
    json_start = text.find("{")
    json_end = text.rfind("}") + 1

    if json_start != -1 and json_end > json_start:
        cleaned = text[json_start:json_end]
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as err:
            logger.error(f"JSON parsing error: {err}")
            logger.debug(f"Failed JSON snippet: {cleaned[:300]}...")
            return {}

    return {}


def generate_therapy_notes(transcript_text: str) -> Dict[str, Any]:
    """
    Generate comprehensive structured clinical therapy notes from session transcript.
    Attempts primary model (gemma-4-31b-it) with automated fallback to gemini-2.5-flash.

    Args:
        transcript_text (str): The conversation transcript between therapist and client.

    Returns:
        Dict[str, Any]: Structured clinical notes dictionary.
    """
    if not client:
        logger.error("Gemini client is not initialized. Please configure GEMINI_API_KEY.")
        return {}

    therapy_prompt = f"""
Act as an experienced clinical psychologist and licensed psychotherapist analyzing a verbatim therapy session transcript.
Review the following conversation between therapist and client, carefully evaluate therapeutic dynamics, and generate structured clinical notes.

**Session Transcript:**
{transcript_text}

**Required JSON Structure:**
{{
  "session_overview": {{
    "summary": "Detailed overview of the main discussion topics and session trajectory.",
    "presenting_concerns": "Client's stated concerns and reasons for seeking therapy.",
    "therapeutic_direction": "Clinical direction and focus of the dialogue."
  }},
  "client_concerns": {{
    "issues": "Specific emotional, cognitive, or relational issues described.",
    "emotional_state": "Observed and reported emotional state during the session."
  }},
  "goals_and_progress": {{
    "short_term_goals": "Immediate actionable goals identified in the session.",
    "long_term_goals": "Overall overarching therapeutic goals.",
    "progress_notes": "Progress made regarding previous interventions or insights."
  }},
  "therapeutic_interventions": {{
    "techniques_used": "Specific therapeutic techniques employed.",
    "rationale": "Clinical reasoning for the chosen interventions."
  }},
  "clients_response": {{
    "engagement_level": "Client's level of participation, openness, and receptiveness.",
    "insights_gained": "Breakthroughs, reframing, or realizations noted by the client.",
    "feedback": "Direct or indirect client feedback regarding interventions."
  }},
  "challenges": {{
    "resistance_noted": "Defensive patterns, resistance, or avoidance behaviors.",
    "transference_issues": "Transference or counter-transference dynamics.",
    "areas_needing_focus": "Clinical areas requiring prioritized attention."
  }},
  "homework_plan": {{
    "assigned_tasks": "Specific exercises, journaling, or behavioral tasks assigned.",
    "focus_areas": "Cognitive/emotional focus areas between sessions.",
    "recommendations": "Additional therapeutic self-care recommendations."
  }},
  "next_session_prompts": {{
    "therapeutic_prompts": [
      "Targeted exploratory questions for the therapist to ask next session aligned with identified techniques."
    ]
  }}
}}

**Clinical Guidance:**
1. Ground all notes strictly in observable transcript evidence; do not invent or assume unmentioned facts.
2. Reference known clinical techniques accurately: {KNOWN_TECHNIQUES}.
3. Maintain objective, professional clinical terminology.
4. Return ONLY the valid JSON object.
"""
    models_to_try = [NOTE_GENERATION_MODEL]
    if FALLBACK_MODEL and FALLBACK_MODEL not in models_to_try:
        models_to_try.append(FALLBACK_MODEL)

    for idx, model_name in enumerate(models_to_try):
        try:
            logger.info(f"Generating clinical notes using model '{model_name}'...")
            response = client.models.generate_content(
                model=model_name,
                contents=therapy_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                ),
            )
            parsed = _extract_json(response.text or "")
            if parsed:
                return parsed
            logger.warning(f"Empty or unparseable JSON received from '{model_name}'.")
        except Exception as e:
            if idx < len(models_to_try) - 1:
                next_model = models_to_try[idx + 1]
                logger.warning(
                    f"Note generation failed with '{model_name}' ({e}); attempting automatic fallback to '{next_model}'."
                )
            else:
                logger.error(f"Error generating therapy notes across all candidate models ({models_to_try}): {e}")

    return {}


def validate_therapy_notes(original_text: str, therapy_notes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cross-validate and verify generated therapy notes against the original transcript
    to prevent hallucinations, omissions, or factual exaggerations.
    Includes automated fallback to fallback model if primary model is unavailable.

    Args:
        original_text (str): The raw conversation transcript.
        therapy_notes (dict): The initial generated therapy notes.

    Returns:
        Dict[str, Any]: Validated and corrected clinical notes dictionary.
    """
    if not client or not therapy_notes:
        return therapy_notes

    validation_prompt = f"""
You are a senior supervising clinical psychologist conducting quality assurance on therapeutic documentation.
Validate the provided clinical notes against the verbatim session transcript.

**Original Conversation Transcript:**
{original_text}

**Draft Clinical Notes:**
{json.dumps(therapy_notes, indent=2)}

**Validation Checklist:**
1. Ensure all notes reflect facts from the conversation without distortion or hallucination.
2. Remove any unsupported assumptions or exaggerated claims.
3. Ensure therapeutic interventions and next-session prompts accurately correspond to what was discussed.
4. Preserve the exact same JSON schema.

Return ONLY the corrected, validated JSON object.
"""
    models_to_try = [NOTE_GENERATION_MODEL]
    if FALLBACK_MODEL and FALLBACK_MODEL not in models_to_try:
        models_to_try.append(FALLBACK_MODEL)

    for idx, model_name in enumerate(models_to_try):
        try:
            logger.info(f"Validating clinical notes against transcript using '{model_name}'...")
            response = client.models.generate_content(
                model=model_name,
                contents=validation_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                ),
            )
            validated = _extract_json(response.text or "")
            if validated:
                return validated
        except Exception as e:
            if idx < len(models_to_try) - 1:
                next_model = models_to_try[idx + 1]
                logger.warning(
                    f"Validation failed with '{model_name}' ({e}); attempting automatic fallback to '{next_model}'."
                )
            else:
                logger.warning(f"Validation step failed ({e}); falling back to initial therapy notes.")

    return therapy_notes


def process_with_gemini(text: Any) -> Dict[str, Any]:
    """
    High-level pipeline: processes session transcript, synthesizes structured
    clinical therapy notes, validates for clinical accuracy, and returns structured data.

    Args:
        text (str or dict): Input transcript text or dictionary containing 'text'.

    Returns:
        Dict[str, Any]: Final validated clinical therapy notes.
    """
    if isinstance(text, dict):
        text_str = text.get("text", "")
    else:
        text_str = str(text)

    if not text_str.strip():
        logger.warning("Empty transcript provided to process_with_gemini.")
        return _empty_notes_structure()

    try:
        # Step 1: Generate initial notes
        raw_notes = generate_therapy_notes(text_str)
        if not raw_notes:
            return _empty_notes_structure()

        # Step 2: Validate against original transcript
        validated_notes = validate_therapy_notes(text_str, raw_notes)
        return validated_notes

    except Exception as e:
        logger.error(f"Error in process_with_gemini pipeline: {e}")
        return _empty_notes_structure()


def _empty_notes_structure() -> Dict[str, Any]:
    """Return fallback empty notes structure."""
    return {
        "session_overview": {
            "summary": "Session documentation unavailable or processing failed.",
            "presenting_concerns": "",
            "therapeutic_direction": ""
        },
        "client_concerns": {},
        "goals_and_progress": {},
        "therapeutic_interventions": {},
        "clients_response": {},
        "challenges": {},
        "homework_plan": {},
        "next_session_prompts": {}
    }