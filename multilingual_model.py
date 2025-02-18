import streamlit as st
import pygame
import base64
from gtts import gTTS
import tempfile
import os
import whisper
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import sounddevice as sd
import numpy as np
import wave
from deep_translator import GoogleTranslator
from langdetect import detect

# Initialize Pygame mixer
pygame.mixer.init()


# Function to get base64 of binary file
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


# Function to set background
def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
    }}
    textarea {{
        caret-color: transparent;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)


set_png_as_page_bg('background_img.jpg')


# Load Whisper model
def load_whisper_model():
    return whisper.load_model("small")


whisper_model = load_whisper_model()

# LangChain setup
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user's queries."),
    ("user", "Question: {question}")
])


# Function to detect if input is a form request
def is_form_request(query):
    form_keywords = ["fill the form", "bank loan application", "job application", "mortgage application"]
    return any(keyword in query.lower() for keyword in form_keywords)


# Function to generate speech file
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        temp_path = tmp_file.name
        tts.save(temp_path)
    return temp_path


# Audio playback functions
def play_audio():
    if st.session_state.audio_file and os.path.exists(st.session_state.audio_file):
        pygame.mixer.music.load(st.session_state.audio_file)
        pygame.mixer.music.play()
        st.session_state.is_playing = True
        st.session_state.is_paused = False


def pause_audio():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.pause()
        st.session_state.is_paused = True


def resume_audio():
    if st.session_state.is_paused:
        pygame.mixer.music.unpause()
        st.session_state.is_paused = False


def stop_audio():
    pygame.mixer.music.stop()
    pygame.mixer.music.unload()
    if st.session_state.audio_file:
        try:
            os.remove(st.session_state.audio_file)
        except Exception as e:
            st.error(f"Error deleting audio file: {e}")
    st.session_state.audio_file = None
    st.session_state.is_playing = False
    st.session_state.is_paused = False


# Function to record and transcribe audio using Whisper
def record_audio(duration=5, samplerate=16000):
    st.info("Recording... Speak now.")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        temp_path = tmp_file.name
        with wave.open(temp_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(audio_data.tobytes())
    return temp_path


def speech_to_text():
    audio_path = record_audio()
    st.info("Transcribing...")
    result = whisper_model.transcribe(audio_path)
    os.remove(audio_path)  # Clean up temp file
    st.success(f"Recognized: {result['text']}")
    return result['text']


# Streamlit UI
st.title("AI Assistant")

# Session state variables
for var in ['audio_file', 'is_playing', 'is_paused', 'response_text']:
    if var not in st.session_state:
        st.session_state[var] = None if var == 'audio_file' else False

# Input options
input_mode = st.radio("Select Input Mode", ["Text Input", "Voice Input"])
input_text = ""

if input_mode == "Text Input":
    input_text = st.text_input("Enter your query")
elif input_mode == "Voice Input":
    if st.button("Speak Now"):
        input_text = speech_to_text()

st.markdown(
    """
    <style>
        .custom-button {
            background-color: red;
            color: white;

            font-family: Silkscreen;
            border-radius: 10px;
            padding: 12px 24px;
            font-size: 15px;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .custom-button:hover {
            background-color: darkred;
        }
    </style>
    """,
    unsafe_allow_html=True
)

if st.markdown('<button class="custom-button">Start Querying</button>', unsafe_allow_html=True):
    if input_text:
        detected_lang = detect(input_text)
        translated_input = GoogleTranslator(source=detected_lang, target="en").translate(
            input_text) if detected_lang != "en" else input_text

        st.info(f"Translated: {translated_input}")

        llm = ChatOllama(model="llama3.2:latest")
        chain = chat_prompt | llm | StrOutputParser()
        response = chain.invoke({'question': translated_input})
        st.session_state.response_text = response  # Store response

# Display response
if st.session_state.response_text:
    st.subheader("Response:")
    st.write(st.session_state.response_text)

# Audio Controls
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("▶️ Play Response") and st.session_state.response_text:
        stop_audio()
        st.session_state.audio_file = text_to_speech(st.session_state.response_text)
        play_audio()
with col2:
    if st.button("⏸️ Pause") and st.session_state.is_playing:
        pause_audio()
with col3:
    if st.button("▶️ Resume") and st.session_state.is_paused:
        resume_audio()
with col4:
    if st.button("⏹️ Stop Speech"):
        stop_audio()