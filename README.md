# TAVI — Talkative Vision Assistant (for the Visually Impaired)

**TAVI** is an AI-powered, multimodal assistant that gives **real-time situational awareness** to visually impaired users. It combines **voice**, **video**, and an **accessible chat-style UI** to support everyday navigation, safety, and confidence.

> Built at the **Hack for Accessibility** hackathon (Google × Code Your Dreams × Deaf Kids Code) by a cross-functional team.

---

## 🚀 Overview

- **Purpose:** Turn audio + video input into clear, spoken feedback.
- **How it works:** A wake word starts listening. TAVI transcribes speech, detects intent, optionally records a short video, understands the scene (caption/OCR), and **speaks back** a concise answer.
- **Modes:**  
  - **Record** → capture frames → **BLIP captions** + **OCR** → LLM summary → **TTS**  
  - **General** → direct LLM reply (no video)

---

## ✨ Key Features

- **🎙 Always-on Voice**
  - Wake-word via **Porcupine** (“Hello Assistant”, “I am done Assistant”)
  - **Whisper** STT → text
  - LLM intent classification (Record / General / Fallback / TAVI-specific)

- **📷 Multimodal Perception**
  - **BLIP** for image captioning
  - **OCR** for reading text in the scene (labels, signs, screens)
  - **LLM** (ChatGroq) for short, safe summaries
  - **pyttsx3** for offline TTS

- **📱 Accessible UI**
  - **Kivy** chat layout (big type, high contrast, screen-reader-friendly)
  - Shows transcribed user query, assistant response, and media previews
  - **Continuous loop:** mic re-activates after each turn

---

## 🛠 Architecture

[User Voice] --(Porcupine wake)--> [Kivy Frontend]
└── audio --> [Whisper STT] --> text
│
▼
[Intent LLM]
┌──────────┴──────────┐
│ │
[General] [Record]
│ │
[LLM response] [Capture frames/videos]
│ ┌───────────┴───────────┐
│ │ │
[TTS] [BLIP captions] [OCR]
│ └──────────┬───────────┘
└──spoken feedback [LLM summary]
│
[TTS]


- **Backend:** FastAPI (`/process_audio`, `/process_video`) orchestrates STT → intent → vision → NLG → TTS  
- **Frontend:** Kivy app (wake-word, audio capture, UI)  
- **Config:** `.env` for keys and endpoints

---

## 📦 Tech Stack

- **Programming:** Python  
- **Frontend:** Kivy  
- **Backend:** FastAPI  
- **AI Models:** Whisper (STT), BLIP (captioning), OCR, ChatGroq LLM (reasoning/summarization), pyttsx3 (TTS)  
- **Wake Word:** Porcupine  
- **Design:** Figma / Framer prototypes for accessible flows

---

## ⚙️ Setup

### 1) Requirements
- Python **3.10+**
- Platform audio permissions (mic + speakers)
- (Optional) GPU for faster vision/LLM

### 2) Clone & install
```bash
git clone <YOUR_REPO_URL>
cd tavi

# (recommended) virtual env
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3) Configure environment

Create a .env at repo root:

# Backends
BACKEND_URL=http://127.0.0.1:8000

# APIs / Keys (use a secrets manager in production)
OPENAI_API_KEY=<optional_if_used>
HUGGINGFACE_API_KEY=<key_for_BLIP_if_required>
GROQ_API_KEY=<key_for_chatgroq>
PORCUPINE_ACCESS_KEY=<porcupine_key>

# Model/runtime options
WHISPER_MODEL=base
BLIP_MODEL=Salesforce/blip-image-captioning-base


Keep keys out of source control.

### ▶️ Run

**Backend** (FastAPI)

uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload


**Frontend** (Kivy)

python frontend/app.py


Say “Hello Assistant” to start.

Speak your request (“What’s in front of me?” / “Read this label.”).

Say “I am done Assistant” to end a turn.

## 🔌 API (Backend)

POST /process_audio
Body: raw audio or base64; returns { text, intent, response_tts }

POST /process_video
Body: frames or a short clip; returns { captions, ocr_text, summary_tts }

GET /health → { status: "ok" }

Exact payloads may vary; see backend/routers/*.py for schemas.

## 🧪 Tips & Testing

Test STT in quiet environments first; adjust mic gain.

Use small models (Whisper base, BLIP base) for laptops.

If TTS stutters, try pyttsx3 voice rate and volume settings.

For OCR, ensure good lighting and steady frames.

## 🗺️ Roadmap

Navigation tie-ins (GPS, beacons)

Wearable support (smart glasses, phone camera relay)

Better streaming (frame batching, VAD for voice)

Multilingual STT/TTS

## 🏆 Acknowledgments

Built at the Hack for Accessibility with support from Google, Code Your Dreams, and Deaf Kids Code.

Thanks to contributors across AI/ML, frontend/backend, UX research, and accessibility testing.
