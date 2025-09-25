from fastapi import FastAPI, UploadFile, Form, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse 
import speech_recognition as sr
from huggingface_hub import login
import torch
from gtts import gTTS
import base64
import uuid
import json
import os, dotenv
import time
import time
from starlette.responses import JSONResponse
from pydantic import BaseModel as BaeseModel

from llm_prompt_query import query_llm  # file: llm_prompt_query.py


app = FastAPI(title="Voice Chatbot with STT and TTS")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

dotenv.load_dotenv()
login(token=os.getenv("LLM_KEY"))

# Ensure audio + history storage
os.makedirs("audio", exist_ok=True)
os.makedirs("history", exist_ok=True)

# Chat history: [(user_input, assistant_response), ...]
chat_history = []
HISTORY_FILE = "chat_history.json"

# initialize the recognizer
r = sr.Recognizer()

device = "cuda" if torch.cuda.is_available() else "cpu"
    

def text_to_speech(text: str, id: str):

    audio_file = f"temp_tts_audio{id}.mp3"
    audio_file_path = f"audio/{audio_file}"
    text_convert = "".join(text[:500])
    tts = gTTS(text=text_convert, lang='en') # only the first 10 lines
    tts.save(audio_file_path)

    try:
        with open(audio_file_path, "rb") as audio:
            base64_data = base64.b64encode(audio.read()).decode('utf-8')
            file_type = "audio/mpeg"  # Adjust based on your file type
            json_data = {
                "fileName": audio_file,
                "fileType": file_type,
                "llmReply": text,
                "audioContent": base64_data}
        
        return JSONResponse(json_data), audio_file_path
    
    except FileNotFoundError:
        return JSONResponse({"error": "Audio file not found"}, status_code=404), ""
    
# Mount the static directory
#app.mount("/", StaticFiles(directory="static", html=True), name="static")

@app.get("/")
async def read_index():
    return FileResponse('index.html')

class UserForm(BaeseModel):

    user_text: str = Form(..., example="Hello, how are you?")
    user_audio: UploadFile = File(None)
    # user_audio: UploadFile = File(None)

@app.post("/chat_full/")
async def chat_service(user_text: str = Form(...), file: UploadFile = File(...)):
    
    print(f"Received user request: {user_text}")
    
    # Save user audio if provided
    if (file and file.size > 0):
        try:
            user_audio_url = None
            file_id = str(uuid.uuid4())
            ext = os.path.splitext(file.filename)[-1] or ".webm"
            file_path = f"audio/user_{file_id}{ext}"

            with open(file_path, "wb") as f:
                f.write(await file.read())
            user_audio_url = f"/{file_path}"
        except Exception as e:
            print("Upload error:", e)
            return {"error": "Failed to save user audio"}

    chat_history.append({"role": "user", "text": user_text})

    llm_response = query_llm(chat_history)

    id_ = str(time.time())
    tts_response, response_audio_file_path = text_to_speech(llm_response, id_)

    # Append to chat history
    chat_history.append({"role": "assistant", "text": llm_response, "audio": response_audio_file_path})
    # Save chat history to file
    log_file_path = f"history/{HISTORY_FILE}"
    with open(log_file_path, "w") as f:
        json.dump(chat_history, f, indent=2)

    return tts_response
