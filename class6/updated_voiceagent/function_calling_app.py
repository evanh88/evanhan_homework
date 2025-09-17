import streamlit as st
from audio_recorder_streamlit import audio_recorder
from streamlit_float import float_init
import speech_recognition as sr
from huggingface_hub import login
from transformers import pipeline
import torch
from gtts import gTTS
import os, dotenv
import time

from llm_prompt_query import query_llm  # file: llm_prompt_query.py

dotenv.load_dotenv()
login(token=os.getenv("LLM_KEY"))

# Chat history: [(user_input, assistant_response), ...]
chat_history = []

# initialize the recognizer
r = sr.Recognizer()

device = "cuda" if torch.cuda.is_available() else "cpu"
    

float_init()

def speech_to_text(file_path: str) -> str:
    
    with sr.AudioFile(file_path) as source:
        # listen for the data (load audio to memory)
        audio_data = r.record(source)
        # recognize (convert from speech to text)
        try:
            text = r.recognize_google(audio_data)
        except sr.UnknownValueError:
            text = "Sorry, I could not understand the audio."
        except sr.RequestError as e:
            text = f"Could not request results; {e}"

        return text

def text_to_speech(text: str, id: str):

    audio_file = f"temp_tts_audio{id}.mp3"
    text_convert = "".join(text[:100])
    tts = gTTS(text=text_convert, lang='en') # only the first 10 lines
    tts.save(audio_file)
    return audio_file   

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Store transcription in session state
    if "draft_text" not in st.session_state:
        st.session_state.draft_text = ""
    if "webm_file_path" not in st.session_state:
        st.session_state.webm_file_path = ""
    # Track if audio is being processed to prevent infinite loops
    if "audio_processed" not in st.session_state:
        st.session_state.audio_processed = False
    # Store the last audio bytes to detect new recordings
    if "last_audio_bytes" not in st.session_state:
        st.session_state.last_audio_bytes = None
    # Track if user has manually edited the text area
    if "text_manually_edited" not in st.session_state:
        st.session_state.text_manually_edited = False


st.title(f"AI Voice Assistant with Function Calling")

initialize_session_state()

# Create footer container for the microphone
footer_container = st.container()
with footer_container:

    audio_bytes = audio_recorder()
    text_input = st.text_area(
        label="Say something...",
        label_visibility="hidden",
        value=st.session_state.draft_text,
        key="draft_text_area",
        height=30
        # key="composer_text_area",
    )

    # Update session state with any manual changes from the text area
    if text_input != st.session_state.draft_text:
        st.session_state.draft_text = text_input
        st.session_state.text_manually_edited = True

    submit = st.button("Submit", type="primary", use_container_width=True)
    
# Write the historical messages to the chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["text"])
        st.audio(message["audio_file"], format="audio/mp3")

# Check if we have new audio that hasn't been processed yet
if audio_bytes and audio_bytes != st.session_state.last_audio_bytes:
    # Reset the processed flag for new audio
    st.session_state.audio_processed = False
    st.session_state.last_audio_bytes = audio_bytes

if audio_bytes and not st.session_state.audio_processed:    # Write the audio bytes to a temporary file
    # This is necessary because Whisper expects a file path
    # and cannot process raw bytes directly.
    
    with st.spinner("Transcribing..."):
        st.session_state.webm_file_path = f"temp_audio_{str(time.time())}.mp3"        
        with open(st.session_state.webm_file_path, "wb") as f:
            f.write(audio_bytes)
        # audio.export(webm_file_path, format="webm")
        
        transcript = speech_to_text(st.session_state.webm_file_path)
        if transcript and not ((transcript.startswith("Sorry, I could not understand the audio.")) |
                                (transcript.startswith("Could not request results;"))):
            # Only update text area if user hasn't manually edited it
            if not st.session_state.text_manually_edited:
                st.session_state.draft_text = transcript
            else:
                # If user has edited, append the transcript to existing text
                if st.session_state.draft_text.strip():
                    st.session_state.draft_text += " " + transcript
                else:
                    st.session_state.draft_text = transcript
        else:
            # Show error message in text area if transcription failed
            if not st.session_state.text_manually_edited:
                st.session_state.draft_text = "Sorry, I could not understand the audio. Please try again."

        # Mark audio as processed to prevent infinite loop
        st.session_state.audio_processed = True

        st.rerun()

if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Processing your request..."):
            final_response = query_llm(st.session_state.messages)
            st.write(final_response) ## not writing the audio response here, it will be posted after TTS is done
        with st.spinner("Generating audio response..."):
            id_ = str(time.time())
            audio_file = text_to_speech(final_response, id_)
            st.audio(audio_file, format="audio/mp3")

    st.session_state.messages.append({"role": "assistant", "text": final_response, "audio_file": audio_file})

if submit and st.session_state.draft_text.strip():
    # Get the current text from the text area (in case user made manual edits)
    current_text = st.session_state.draft_text.strip()
    st.session_state.messages.append({"role": "user", "text": current_text, "audio_file": st.session_state.webm_file_path})
 
    with st.chat_message("user"):
        st.write(st.session_state.draft_text)
        st.audio(st.session_state.webm_file_path, format="audio/mp3")
        
    st.session_state.draft_text = ""
    st.session_state.text_manually_edited = False  # Reset the manual edit flag
        
    st.rerun()


# Float the footer container to the bottom of the page
footer_container.float("bottom: 0rem;")
