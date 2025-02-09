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

def run_script(input_path, filler_words):
    """Run video_toolv3.py with the provided input path and filler words."""
    try:
        # Pass the input path and filler words as arguments
        process = subprocess.Popen(
            ["python", "video_toolv3.py", input_path, "--filler-words", ",".join(filler_words)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return process
    except Exception as e:
        st.error(f"Error starting subprocess: {e}")
        return None

# Processing button
if st.button("Process Video"):
    if uploaded_file is not None:
    # Save the uploaded file temporarily
        save_path = os.path.join("temp_videos", uploaded_file.name)
        os.makedirs("temp_videos", exist_ok=True)  # Ensure the directory exists

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())  # Save file to disk

        st.success(f"File saved: {save_path}")
        input_path = save_path
    elif video_url:
        input_path = video_url
    else:
        st.error("Please upload a file or provide a video URL.")
        st.stop()
    with st.spinner("Processing..."):
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
   
    # Run the backend script
    process = run_script(input_path, filler_words)

    if process is None:
        st.error("Failed to start the processing script.")
        st.stop()

    # Monitor progress
    while process.poll() is None:  # While the process is running
        if os.path.exists("progress.txt"):
            with open("progress.txt", "r") as f:
                progress = int(f.read().strip())
            progress_bar.progress(progress)
            status_text.text(f"Progress: {progress}%")
        time.sleep(1)  # Check progress every second

    # Final update
    progress_bar.progress(100)
    status_text.text("Processing complete!")

    # Check the result of the process
    stdout, stderr = process.communicate()
    if process.returncode == 0:
        st.success("Processing complete!")
        st.write(stdout)  # Display the output of video_toolv3.py
    else:
        st.error(f"Processing failed. Error: {stderr}")