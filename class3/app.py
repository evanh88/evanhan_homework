import dotenv
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from streamlit_float import *
import speech_recognition as sr
from huggingface_hub import login
from transformers import pipeline
import torch
from gtts import gTTS
import os
import time

dotenv.load_dotenv()
login(token=os.getenv("LLM_KEY"))
llm = pipeline("text-generation", model="meta-llama/Llama-3.2-3B-Instruct", device=0 if torch.cuda.is_available() else -1)

# initialize the recognizer
r = sr.Recognizer()

SYSTEM_PROMPT = {
    "role": "system",
    "text": "You are a helpful assistant. The user will provide you with questions transcribed from audio input. \
You job is to have a friendly conversation with the user and answer their questions."}

device = "cuda" if torch.cuda.is_available() else "cpu"


float_init()

def speech_to_text(file_path: str) -> str:
    
    with sr.AudioFile(file_path) as source:
        # listen for the data (load audio to memory)
        audio_data = r.record(source)
        # recognize (convert from speech to text)
        text = r.recognize_google(audio_data)
        return text

def get_answer(user_text):
    # Construct prompt from system prompt and history
    prompt = f"{SYSTEM_PROMPT['role']}: {SYSTEM_PROMPT['text']}\n"
    for turn in st.session_state.messages[-4:]:
        if turn["role"] != "system":
            prompt += f"{turn['role']}: {turn['text']}\n"
            
    outputs = llm(prompt, max_new_tokens=100)
    bot_response = outputs[0]["generated_text"]  
    return bot_response

def text_to_speech(text: str, id: str):

    audio_file = f"temp_tts_audio{id}.mp3"
    tts = gTTS(text=text, lang='en')
    tts.save(audio_file)
    return audio_file   

def initialize_session_state():
        if "messages" not in st.session_state:
            st.session_state.messages = []

st.title(f"AI Voice Assistant-{device}")

initialize_session_state()

# Create footer container for the microphone
footer_container = st.container()
with footer_container:
    audio_bytes = audio_recorder()
        
# Write the historical messages to the chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["text"])
        st.audio(message["audio_file"], format="audio/mp3")

if audio_bytes:
    # Write the audio bytes to a temporary file
    # This is necessary because Whisper expects a file path
    # and cannot process raw bytes directly.
    with st.spinner("Transcribing..."):
        webm_file_path = f"temp_audio_{str(time.time())}.mp3"        
        with open(webm_file_path, "wb") as f:
            f.write(audio_bytes)
        # audio.export(webm_file_path, format="webm")
        
        transcript = speech_to_text(webm_file_path)
        if transcript:
            st.session_state.messages.append({"role": "user", "text": transcript, "audio_file": webm_file_path})
            with st.chat_message("user"):
                st.write(transcript)
                st.audio(webm_file_path, format="audio/mp3")

if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Processing your request..."):
            final_response = get_answer(st.session_state.messages)
            st.write(final_response) ## not writing the response here, it will be handled by TTS
        with st.spinner("Generating audio response..."):
            id_ = str(time.time())
            audio_file = text_to_speech(final_response, id_)

            st.audio(audio_file, format="audio/mp3")

    st.session_state.messages.append({"role": "assistant", "text": final_response, "audio_file": audio_file})

# Float the footer container to the bottom of the page
footer_container.float("bottom: 0rem;")