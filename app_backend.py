import os
import uuid
import logging
import asyncio

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from processing import SurroundingAwarenessProcessor
from audio_processing import AudioProcessing, GLOBAL_TEXT_SUMMARY

# Set up logging for the API. Only errors will be printed.
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

app = FastAPI(title="Surrounding Awareness API")

# Enable CORS for future integration (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the processors (video and audio)
processor = SurroundingAwarenessProcessor()
audio_processor = AudioProcessing()

@app.post("/process_video/")
async def process_video(file: UploadFile = File(...)):
    """
    Accept a video file, process it through the pipeline, and return the generated text summary 
    and audio file (MP3). Also, update the global text summary so that the audio processing 
    endpoint has access to the latest video summary.
    """
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_id = str(uuid.uuid4())
    temp_file_path = os.path.join(temp_dir, f"{file_id}_{file.filename}")
    
    try:
        # Save the uploaded video to a temporary location.
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        loop = asyncio.get_event_loop()
        video_process_result = await loop.run_in_executor(None, processor.process_video, temp_file_path)
        combined_text = video_process_result.get("combined_text", "")
        if not combined_text:
            raise HTTPException(status_code=500, detail="Failed to extract content from video.")

        llm_summary = await loop.run_in_executor(None, processor.generate_llm_summary, combined_text)
        if not llm_summary:
            raise HTTPException(status_code=500, detail="LLM summarization failed.")
        
        audio_output_path = os.path.join(temp_dir, f"{file_id}_output.mp3")
        audio_success = await loop.run_in_executor(None, processor.generate_audio, llm_summary, audio_output_path)
        if not audio_success:
            raise HTTPException(status_code=500, detail="Audio generation failed.")

        # Update global text summary store; each new video overwrites the previous summary.
        GLOBAL_TEXT_SUMMARY["latest"] = llm_summary

        response = {
            "text_summary": llm_summary,
            "audio_file": f"/download_audio/{file_id}_output.mp3"
        }
        return response
    except Exception as e:
        logger.error(f"Error in processing video API: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
    finally:
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        except Exception as e:
            logger.warning(f"Error cleaning up temporary file: {e}")

@app.post("/process_audio/")
async def process_audio(file: UploadFile = File(...)):
    """
    Accept an audio file, process it through the audio processing pipeline, and return:
    - data1: The intent recognition JSON
    - data2: The generated text response
    - data3: The path to the TTS-generated audio file
    - transcript: The STT-generated transcript (for debugging)
    """
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_id = str(uuid.uuid4())
    temp_file_path = os.path.join(temp_dir, f"{file_id}_{file.filename}")

    try:
        # Save the uploaded audio file.
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        loop = asyncio.get_event_loop()
        audio_result = await loop.run_in_executor(None, audio_processor.process_audio, temp_file_path)
        return audio_result
    except Exception as e:
        logger.error(f"Error in processing audio API: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
    finally:
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        except Exception as e:
            logger.warning(f"Error cleaning up temporary file: {e}")

@app.get("/download_audio/{audio_filename}")
async def download_audio(audio_filename: str):
    """
    Endpoint to download the generated audio file.
    """
    file_path = os.path.join("temp_uploads", audio_filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path=file_path, filename=audio_filename, media_type='audio/mpeg')
