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
import re
from starlette.responses import JSONResponse
from pydantic import BaseModel as BaeseModel
from markdown import markdown
from bs4 import BeautifulSoup

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
    
def looks_like_markdown(text: str) -> bool:
    markdown_patterns = [
        r'^#{1,6}\s',                  # headings like "# Heading"
        r'\*\*.*?\*\*',                # bold **text**
        r'\*.*?\*',                    # italics *text*
        r'`{1,3}.*?`{1,3}',            # inline or fenced code
        r'^\s*[-+*]\s',                # unordered list
        r'^\s*\d+\.\s',                # ordered list
        r'\[.*?\]\(.*?\)',            # links [text](url)
        r'^```',                      # code blocks
        r'\|.*?\|',                   # tables
    ]

    for pattern in markdown_patterns:
        if re.search(pattern, text, re.MULTILINE):
            return True
    return False

def markdown_to_plain_text(md_text):
    html = markdown(md_text)
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")  # optional: separator=" " or "\n"

def text_to_speech(text: str, id: str):

    audio_file = f"temp_tts_audio{id}.mp3"
    audio_file_path = f"audio/{audio_file}"
    text_convert = "".join(text[:500])  # only convert part of the text
    tts = gTTS(text=text_convert, lang='en') # only the first 10 lines
    tts.save(audio_file_path)

    return audio_file_path


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

    # Call to query LLM
    llm_response = query_llm(chat_history)

    llm_response_raw_text = ""
    if looks_like_markdown(llm_response):
        llm_response_plain_text = markdown_to_plain_text(llm_response)
    else:
        llm_response_plain_text = llm_response

    # Convert raw text to speech
    id_time = str(time.time())
    audio_file_path = text_to_speech(llm_response_plain_text, id_time)

    # Build response
    try:
        with open(audio_file_path, "rb") as audio:
            base64_data = base64.b64encode(audio.read()).decode('utf-8')
            file_type = "audio/mpeg"  # Adjust based on your file type
            json_data = {
                "filePath": audio_file_path,
                "fileType": file_type,
                "llmReply": llm_response,  # Use the text format returned by LLM here. Client will convert Markdown text it to HTML
                "audioContent": base64_data}
    except FileNotFoundError:
        return JSONResponse({"error": "Audio file not found"}, status_code=404), ""

    # Append to chat history
    chat_history.append({"role": "assistant", "text": llm_response_plain_text, "audio": audio_file_path})
    # Save chat history to file
    log_file_path = f"history/{HISTORY_FILE}"
    with open(log_file_path, "w") as f:
        json.dump(chat_history, f, indent=2)

    # Return data to client
    return JSONResponse(json_data)
