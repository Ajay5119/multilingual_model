import streamlit as st
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

# Load Whisper model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("small")

whisper_model = load_whisper_model()

# LangChain setup
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user's queries."),
    ("user", "Question: {question}")
])

# Function to generate speech file
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        temp_path = tmp_file.name
        tts.save(temp_path)
    return temp_path

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
st.title("AI Multilingual Assistant")

# Input options
input_mode = st.radio("Select Input Mode", ["Text Input", "Voice Input"])
input_text = ""

if input_mode == "Text Input":
    input_text = st.text_input("Enter your query")
elif input_mode == "Voice Input":
    if st.button("🎤 Speak Now"):
        input_text = speech_to_text()

if st.button("🚀 Start Querying"):
    if input_text:
        detected_lang = detect(input_text)
        translated_input = GoogleTranslator(source=detected_lang, target="en").translate(
            input_text) if detected_lang != "en" else input_text

        st.info(f"Translated: {translated_input}")

        llm = ChatOllama(model="llama3.2:latest")
        chain = chat_prompt | llm | StrOutputParser()
        response = chain.invoke({'question': translated_input})
        
        st.subheader("Response:")
        st.write(response)

        # Convert text to speech
        audio_file = text_to_speech(response)
        st.audio(audio_file, format="audio/mp3")

if __name__ == "__main__":
    st.write("App is running...")
