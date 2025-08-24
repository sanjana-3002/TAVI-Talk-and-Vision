import os
import json
import logging
import tempfile
from dotenv import load_dotenv

import pyttsx3
import requests  # in case needed
from openai import OpenAI
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Set up logging (only errors and warnings are shown)
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Global text summary store
GLOBAL_TEXT_SUMMARY = {"latest": ""}

class AudioProcessing:
    def __init__(self):
        try:
            self.openai_api_key = os.getenv('OPENAI_API_KEY')
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables.")
            # Initialize OpenAI client (for Whisper STT and intent recognition)
            self.openai_client = OpenAI(api_key=self.openai_api_key)
            self.intent_model = "gpt-4o-mini"
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

        try:
            self.groq_api_key = os.getenv('GROQ_API_KEY')
            if not self.groq_api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables.")
            self.llm = ChatGroq(
                temperature=0.2, 
                model_name="llama-3.3-70b-versatile",
                api_key=self.groq_api_key
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChatGroq LLM: {e}")
            raise

        try:
            # Initialize pyttsx3 engine for TTS
            self.tts_engine = pyttsx3.init()
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            raise

    def speechtotext(self, audio_path: str) -> str:
        """
        Convert the audio file to text using OpenAI's Whisper API.
        """
        try:
            with open(audio_path, "rb") as audio_file:
                transcript_response = self.openai_client.audio.translations.create(
                    model="whisper-1",
                    file=audio_file
                )
            # Access the text attribute directly
            transcript = transcript_response.text
            return transcript
        except Exception as e:
            logger.error(f"Error in speech-to-text conversion: {e}")
            return ""

    def intent_recognition(self, processed_text: str) -> dict:

        """
        Identify the user's intent using OpenAI's model.
        Returns a JSON object with boolean flags for intents.
        """
        # Default system prompt (used if no video summary is available)
        default_system_prompt = """
            You are an AI designed to identify the user's intent from transcribed audio.

            Given a transcribed sentence, classify the user's intent into exactly one of the following categories:

            1. "Record" – The user is requesting to record a video.  
            Example: "Start recording", "Can you record a video for me?", "Begin capturing now."

            2. "General" – The user is asking a general knowledge or informational question that is unrelated to their environment.  
            Example: "What is photosynthesis?", "Tell me about climate change."

            3. "Fallback" – The user's request is out of context or unrelated to the system's domain, such as booking flights or managing bank accounts.  
            Example: "Can you book a flight to Australia?", "Transfer money to my account."

            4. "Tavi" – The user is asking a question about their **immediate surroundings**, **based on a video they have recorded** or are currently recording.  
            These questions usually refer to real-world visual context such as nearby locations, people, objects, or navigation.  
            Example: "Are there any places to eat nearby?", "Who is standing near the bus stop?", "What's in front of me?"

            You must classify the input into **only one** of the above intents. Set that intent's value to `true`, and set all others to `false`.  
            Return your output in the following JSON format:

            {
                "Record": false,
                "General": false,
                "Fallback": false,
                "Tavi": true
            }

            Strictly return only the JSON object—no explanation, preamble, or extra text.
            """

        
        # Check for a global video summary
        if GLOBAL_TEXT_SUMMARY.get("latest", "").strip():
            video_summary = GLOBAL_TEXT_SUMMARY["latest"]
            custom_system_prompt = f"""
            You are an AI designed to identify the user's intent from audio transcript and by using context from a previously generated video summary.
            Here is the video summary: {video_summary}

            Based on the transcribed text and video summary provided to you, classify the transcribed text into exactly one of the following intents:

            1. "Record" – The user is asking to record a video.
            Example: "Start recording", "Can you record a video for me?", "Begin capturing now."
            2. "General" – The user is asking a general knowledge or informational question.
            Example: "What is photosynthesis?", "Tell me about climate change."
            3. "Fallback" – The user is asking something completely out of the assistant's capabilities.
            Example: "Can you book a flight to Australia?", "Transfer money to my account."
            4. "Tavi" – The user is asking for a query about the video they have recorded of their surroundings.
            Example: "Are there any places to eat nearby?", "Who is standing near the bus stop?", "What's in front of me?"

            Use the video summary context to help determine if the user's query is about their surroundings.
            You must classify the input into only one of the above intents. Set that intent's value to true, and set all others to false.
            Return your output as a JSON object in the following format:

            {{
                "Record": false,
                "General": false,
                "Fallback": false,
                "Tavi": true
            }}
            Strictly return only the JSON object—no explanation, preamble, or extra text. 
            """
        else:
            custom_system_prompt = default_system_prompt

        try:
            response = self.openai_client.chat.completions.create(
                model=self.intent_model,
                messages=[
                    {"role": "system", "content": custom_system_prompt},
                    {"role": "user", "content": processed_text}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            # Access the content of the returned message using dot notation
            intent_json = json.loads(response.choices[0].message.content)
            return intent_json
        except Exception as e:
            logger.error(f"Error during intent recognition: {e}")
            return {}


    def text_to_speech(self, text: str, output_path: str) -> bool:
        """
        Convert the provided text to speech using pyttsx3 and save as an audio file.
        """
        try:
            # For empty text, generate a silent file by simply creating an empty file.
            if not text.strip():
                with open(output_path, "wb") as f:
                    f.write(b"")
                return True

            self.tts_engine.save_to_file(text, output_path)
            self.tts_engine.runAndWait()
            return True
        except Exception as e:
            logger.error(f"Error generating TTS audio: {e}")
            return False

    def process_audio(self, audio_file_path: str) -> dict:
        """
        Main processing function for audio:
            1. Convert audio file to text using Whisper (STT).
            2. Use intent recognition to classify the transcript.
            3. Based on the intent, generate response text (data2):
                a. Record intent: return empty response.
                b. General intent: send transcript to ChatGroq LLM for a brief answer.
                c. Fallback intent: return a fixed fallback message.
                d. Tavi intent: if a global text summary exists, combine it with transcript to query ChatGroq; else, return a fallback message.
            4. Convert response text to audio (TTS) to generate data3.
            5. Return a dictionary with data1 (intent), data2 (text response), and data3 (audio file path).
        """
        # Step 1: Speech-to-Text conversion
        audio_transcript = self.speechtotext(audio_file_path)

        # Step 2: Intent Recognition
        data1 = self.intent_recognition(audio_transcript)

        # Initialize data2 as empty text and define path for TTS audio output.
        data2 = ""
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        audio_output_path = os.path.join(temp_dir, f"{os.path.basename(audio_file_path)}_response.mp3")

        # Step 3: Branch based on recognized intent.
        try:
            # (a) Record intent: return empty response text and an empty audio file.
            if data1.get("Record"):
                data2 = ""
                with open(audio_output_path, "wb") as f:
                    f.write(b"")
            # (b) General intent: Use transcript to query ChatGroq for a brief answer.
            elif data1.get("General"):
                messages = [
                    ("system", "Provide a concise answer between 30 and 40 words for the following query:"),
                    ("human", audio_transcript)
                ]
                response = self.llm.invoke(messages)
                data2 = response.content.strip()
                self.text_to_speech(data2, audio_output_path)
            # (c) Fallback intent: Return fallback message.
            elif data1.get("Fallback"):
                data2 = "I am sorry, I cannot help you with this request. Could you try asking again?"
                self.text_to_speech(data2, audio_output_path)
            # (d) Tavi intent: Query about the video surroundings.
            elif data1.get("Tavi"):
                if GLOBAL_TEXT_SUMMARY.get("latest", "").strip():
                    messages = [
                        (
                            "system",
                            "You are an AI assistant helping a user understand their surroundings from a video. Based on the provided summary of the user's environment, answer the user's query with a brief and clear response **only if the information is available in the video summary**. "
                            "If the summary does not contain the relevant information, respond with: \"I'm sorry, I couldn't find what you're looking for in the recorded video.\" "
                            "Your response must be a single sentence, with no preamble or additional explanation."
                        ),
                        ("human", f"Video Summary of user surroundings: {GLOBAL_TEXT_SUMMARY['latest']} User Query about surroundings: {audio_transcript}")
                    ]
                    response = self.llm.invoke(messages)
                    data2 = response.content.strip()
                    self.text_to_speech(data2, audio_output_path)
                else:
                    data2 = "Please record video before asking questions about your surroundings."
                    self.text_to_speech(data2, audio_output_path)
            else:
                # If none of the expected intents is true, return a fallback.
                data2 = "I am sorry, I cannot help you with this request. Could you try asking again?"
                self.text_to_speech(data2, audio_output_path)
        except Exception as e:
            logger.error(f"Error during intent-based processing: {e}")
            data2 = "I am sorry, something went wrong processing your request."
            self.text_to_speech(data2, audio_output_path)

        # Step 5: Prepare the final output.
        #data3 = audio_output_path  # Path to the generated audio file.
        audio_filename = os.path.basename(audio_output_path)
        data3 = f"/download_audio/{audio_filename}"
        return {
            "data1": data1,         # The intent recognition JSON.
            "data2": data2,         # The generated text response.
            "data3": data3,         # The path to the TTS-generated audio file.
            "transcript": audio_transcript  # The STT-generated transcript (for debugging).
        }
