import streamlit as st
import os
import tempfile
from moviepy.editor import VideoFileClip
import whisper
import librosa
from pydub import AudioSegment
import numpy as np
import requests
import subprocess
import threading
import time

os.chdir(os.path.dirname(__file__))  # Ensure script runs in the correct directory

# Set up page title
st.set_page_config(page_title="Automated Video Editor", layout="wide")

# Title and description
st.title("Automated Video Editor")
st.markdown("This tool allows you to process a video by removing user-selected filler words and silent portions.")

# Upload a file or provide a URL
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])
video_url = st.text_input("Or enter a video URL:")

# Get user input for filler words
filler_words_input = st.text_input("Enter filler words to remove (comma-separated):")
filler_words = [word.strip().lower() for word in filler_words_input.split(",") if word.strip()]

def run_script():
    subprocess.run(["python", "video_toolv3.py"], check=True)

if uploaded_file is not None:
    # Save the uploaded file temporarily
    save_path = os.path.join("temp_videos", uploaded_file.name)
    os.makedirs("temp_videos", exist_ok=True)  # Ensure the directory exists

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())  # Save file to disk

    st.success(f"File saved: {save_path}")

# Processing button
if st.button("Process Video"):
    with st.spinner("Processing..."):
        # Create a progress bar
        progress_bar = st.progress(0)
        
        # Simulate processing with a progress update
        for percent_complete in range(100):
            time.sleep(0.1)  # Simulate a time-consuming task
            progress_bar.progress(percent_complete + 1)

        # Run video_toolv3.py with the saved file path as an argument
        result = subprocess.run(["python", "video_toolv3.py", save_path], capture_output=True, text=True)
        
        # Display any errors if the script fails
        if result.returncode != 0:
            st.error(f"Error: {result.stderr}")
        else:
            st.success("Processing complete!")