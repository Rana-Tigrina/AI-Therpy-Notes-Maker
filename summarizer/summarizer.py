import json
import logging
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables once
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

# Initialize the new GenAI client
client = genai.Client(api_key=api_key)

# Example of a curated list reference
KNOWN_TECHNIQUES = [
    "Cognitive-Behavioral Therapy (CBT)",
    "Solution-Focused Brief Therapy (SFBT)",
    "Behavioral Techniques",
    "Person-Centered Therapy",
    "Stress Management Techniques",
    "Narrative Therapy",
    "Psychoeducation",
    "Strength-Based Approach",
    "Socratic Questioning (Cognitive Therapy)",
    "Motivational Interviewing (MI)",
    "Mindfulness-Based Interventions",
    "Emotion-Focused Therapy (EFT)",
    "Open-Ended Questioning",
    "Reflective Questions",
    "Grounding and Relaxation Techniques",
    "Values Clarification",
    "Problem-Solving Therapy",
    "Acceptance and Commitment Therapy (ACT)",
    # ...and so on...
]

def grammar_agent(text_str: str) -> str:
    """
    Correct grammar, fill missing words, and ensure text consistency without altering the original meaning.
    Specifically tailored for medical transcripts containing both Hindi and English.
    """
    grammar_prompt = f"""
    You are a highly accurate grammar correction tool specialized in medical transcripts that may contain both Hindi and English.
    Perform the following tasks:
    1. Correct all grammatical errors.
    2. Fill in missing words or sentences based on the context.
    3. Ensure that the corrected text maintains the original meaning without adding any new information.
    4. Preserve the bilingual nature of the transcript without altering the language of each segment.

    Original Transcript:
    {text_str}

    Corrected Transcript:

    IMPORTANT ADDITIONAL GUIDANCE:
    - Only correct grammar where context is clear, do not add or exaggerate facts not present in the text.
    - If any part of the sentence is ambiguous or incomplete, infer meaning conservatively without inventing details.
    """
    grammar_response = client.models.generate_content(
        model='gemma-4-31b-it',
        contents=grammar_prompt,
    )
    corrected_text = grammar_response.text.strip()

    if not corrected_text:
        logging.error("Grammar agent returned empty response.")
        return text_str  # Return original text if correction fails

    return corrected_text

def generate_therapy_notes(corrected_text: str) -> dict:
    """
    Generates detailed therapy notes including identified techniques and next session prompts.

    Args:
        corrected_text (str): The grammatically corrected conversation text.

    Returns:
        dict: Structured therapy notes.
    """
    therapy_prompt = f"""
    Act as an experienced psychotherapist analyzing a therapy session transcript. Review the following conversation 
    between a therapist and client, carefully noting the therapeutic dynamics and generate detailed session notes:

    {corrected_text}

     Analyze the conversation and provide a structured therapeutic note in the following JSON format:
    {{ ... }}

    {{
      "session_overview": {{
        "summary": "Provide a Detailed overview of the main topics and session focus.",
        "presenting concerns": "Detail the client's initial concerns or reasons for seeking therapy.",
        "therapeutic direction": "Describe the overall direction of the therapeutic discussion."
      }},
      "client_concerns": {{
        "issues": "Elaborate on the specific issues faced by the client.",
        "emotional state ": "Describe the client's emotional state during the session."
      }},
      "goals_and_progress": {{
        "short term goals": "Immediate objectives identified in numbered or bullet list",
        "long term goals": "Overall therapeutic goals in numbered or bullet list",
        "progress notes": "Progress on previous goals if applicable"
      }},
      "therapeutic_interventions": {{
        "techniques used": "Specific therapeutic techniques employed",
        "rationale": "Reasoning for chosen interventions"
      }},
      "clients_response": {{
        "engagement level": "Assess the client's participation and receptiveness.",
        "insights gained": "Note any key realizations or breakthroughs.",
        "feedback": "Record the client's feedback on the interventions."
      }},
      "challenges": {{
        "resistance noted": "Identify any resistance or defensive patterns.",
        "transference issues":  "Highlight notable transference or countertransference.",
        "areas needing focus": "List of Challenges requiring additional attention"
      }},
      "homework_plan": {{
        "assigned tasks": "Specific homework or exercises assigned",
        "focus areas": "Areas to work on before next session",
        "recommendations": "Additional therapeutic recommendations"
      }},
      "next_session_prompts": {{
      "name of the technique": ["list next session question which the doctor can ask based on the todays sessions that aligned to techniques actually used in session and make sure questions must be relevant"],
      }}
    }}

    Ensure the analysis is professional, objective, and focuses on observable behaviors and therapeutic interactions.

    Known techniques (for reference only, do not invent new ones or hallucinate):
    {KNOWN_TECHNIQUES}

    IMPORTANT (for identfied_techniques and next_session_prompts):
    1. Refer only to known, relevant therapy techniques from a curated list.
    2. Include techniques in "identified_techniques" strictly if they match the conversation.
    3. Base "next_session_prompts" on the identified techniques without introducing new or unmentioned methods.
    4. Avoid hallucinating techniques; if none apply, leave 'identified_techniques' empty.

    EXTRA CAUTION:
    - Remain strictly true to the conversation; do not assume or infer substantial unmentioned details.
    - If certain information is missing or unclear, indicate that it is not mentioned instead of filling hypothetical data.

    FORMATTING RULES:
    1. Output ONLY the JSON object, nothing else
    2. Use ONLY double quotes for strings and keys
    3. NO trailing commas
    4. NO comments or explanations
    5. COMPLETE all fields, If data is missing, leave placeholders or note that it's not mentioned rather than creating facts.
    6. STOP once the JSON object is complete
    """
    try:
        therapy_response = client.models.generate_content(
            model='gemma-4-31b-it',
            contents=therapy_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        logging.info(f"Gemini API therapy notes response: {therapy_response.text}")

        response_text = therapy_response.text.strip()
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            response_text = response_text[json_start:json_end]
            try:
                therapy_notes = json.loads(response_text)
                return therapy_notes
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error in therapy notes: {str(e)}")
                logging.debug(f"Invalid JSON response: {response_text}")
                return {}
        else:
            logging.error("No JSON content found in therapy notes response.")
            logging.debug(f"Response received: {response_text}")
            return {}
    except Exception as e:
        logging.error(f"Error generating therapy notes: {e}")
        return {}

def validate_therapy_notes(original_text: str, therapy_notes: dict) -> dict:
    """
    Validates and corrects the therapy notes against the original conversation text.

    Args:
        original_text (str): The original conversation text between therapist and client.
        therapy_notes (dict): The generated therapy notes.

    Returns:
        dict: Validated and corrected therapy notes.
    """
    validation_prompt = f"""
    You are an expert psychotherapist and analyst. Perform validation of the therapeutic notes against the original conversation.

    **Original Conversation:**
    {original_text}

    **Therapeutic Notes:**
    {json.dumps(therapy_notes, indent=2)}

    **Validation Tasks:**
    1. Identify any inaccuracies or omissions in the therapeutic notes compared to the original conversation.
    2. Remove any exaggerations, unsupported assumptions, or newly introduced content.
    3. Provide necessary corrections to ensure the therapeutic notes accurately reflect the original conversation.
    4. Ensure that 'identified_techniques' and 'next_session_prompts' are accurately reflected based on the conversation.
    5. If data is missing, leave placeholders or note that it's not mentioned rather than creating facts.

    Ensure:
    1. 'identified_techniques' accurately match the conversation and do not include irrelevant methods.
    2. Any techniques incorrectly listed are removed or corrected.
    3. 'next_session_prompts' correspond precisely to the corrected techniques.


    **Your Validation and Corrections:**
    Return the corrected therapeutic notes in the same JSON format.    
    """

    try:
        validation_response = client.models.generate_content(
            model='gemma-4-31b-it',
            contents=validation_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        logging.info(f"Gemini API validation response: {validation_response.text}")

        response_text = validation_response.text.strip()
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            response_text = response_text[json_start:json_end]
            try:
                validated_notes = json.loads(response_text)
                return validated_notes
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error in validation: {str(e)}")
                return therapy_notes  # Return original if validation fails
        else:
            logging.error("No JSON content found in validation response.")
            return therapy_notes  # Return original if no JSON found
    except Exception as e:
        logging.error(f"Error validating therapy notes: {e}")
        return therapy_notes

def process_with_gemini(text: str) -> dict:
    """
    Processes text using Gemini API with grammar correction, generates therapy notes including techniques and prompts,
    validates the notes, and returns the final structured therapy notes.

    Args:
        text (str): The input text to process.

    Returns:
        dict: The final structured therapy notes after validation.
    """
    try:
        if isinstance(text, dict):
            text_str = text.get("text", "")
        elif isinstance(text, pd.DataFrame):
            text_str = "\n".join(f"{row.get('speaker_role', 'unknown')}: {row.get('text', '')}" 
                                 for _, row in text.iterrows())
        else:
            text_str = str(text)

        if not api_key:
            logging.error("GEMINI_API_KEY not found in environment variables")
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        # Step 1: Grammar Correction
        # corrected_text = grammar_agent(text_str)

        # Step 2: Generate Therapy Notes (including techniques and prompts)
        # therapy_notes = generate_therapy_notes(corrected_text)
        
        therapy_notes = generate_therapy_notes(text_str)
        print('\n\nNotes generated\n\n\n')

        # Step 3: Validate Therapy Notes
        validated_therapy_notes = validate_therapy_notes(text_str, therapy_notes)
        print('\n\n\nNotes validated\n\n\n')

        return validated_therapy_notes
        
        # return therapy_notes

    except Exception as e:
        logging.error(f"Gemini API error: {e}")
        return {
            "session_overview": {},
            "client_concerns": {},            
            "goals_and_progress": {},            
            "therapeutic_interventions": {},
            "clients_response": {},
            "challenges": {},
            "homework_plan": {},
            "identified_techniques": [],
            "next_session_prompts": []
        }
    

# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) != 2:
#         print("Usage: python process_with_gemini.py <path_to_transcript_file>")
#         sys.exit(1)

#     transcript_file_path = sys.argv[1]
    
#     if not os.path.isfile(transcript_file_path):
#         print(f"File not found: {transcript_file_path}")
#         sys.exit(1)

#     with open(transcript_file_path, 'r') as file:
#         input_text = file.read()

#     result = process_with_gemini(input_text)
#     print(json.dumps(result, indent=2))

def process_with_local(transcript: str, model_name: str = "llama") -> dict:
    """
    Summarizes the transcript using a local Ollama instance.
    """
    import requests
    url = "http://localhost:11434/api/chat"
    prompt = f"""
You are an expert clinical psychologist and AI assistant. Your task is to analyze the following transcription of a therapy session and generate structured, clinical therapy notes.

The notes MUST be structured as a JSON object containing exactly these 8 keys:
1. "session_overview": A summary of the session flow, dynamic, and tone. (Format: list of sentences/points or a single paragraph string)
2. "client_concerns": Primary issues, feelings, or thoughts the client presented. (Format: list of strings)
3. "goals_and_progress": Treatment goals discussed, and any progress made towards them. (Format: list of strings)
4. "therapeutic_interventions": Techniques, strategies, or interventions used by the therapist. (Format: list of strings)
5. "clients_response": How the client reacted to interventions and their level of engagement. (Format: list of strings or single paragraph string)
6. "challenges": Difficulties, resistances, or barriers observed. (Format: list of strings)
7. "homework_plan": Specific tasks or exercises assigned for the client to complete before the next session. (Format: list of strings or single paragraph string)
8. "next_session_prompts": Questions, topics, or focus areas for the next session. (Format: list of strings)

Respond ONLY with the JSON object. Do not include markdown code block syntax (like ```json ... ```). Make sure it is valid parseable JSON.

Therapy Session Transcript:
{transcript}
"""
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "format": "json"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        content = result.get("message", {}).get("content", "").strip()
        return json.loads(content)
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Local LLM server (Ollama) is not running on http://localhost:11434. Please start Ollama or select the Gemini model."
        )
    except Exception as e:
        raise RuntimeError(f"Local LLM processing failed: {str(e)}")