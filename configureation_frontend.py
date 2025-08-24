# config.py
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env if present

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
HF_API_KEY = os.getenv("HF_API_KEY")
# You can add other configuration values here as needed.
PORCUPINE_KEY = os.getenv("PORCUPINE_KEY")
