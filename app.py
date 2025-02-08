import streamlit as st
import os
import tempfile
from moviepy.editor import VideoFileClip
import whisper
import librosa
from pydub import AudioSegment
import numpy as np
import requests

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

# Processing button
if st.button("Process Video"):
    if not uploaded_file and not video_url:
        st.error("Please upload a video file or provide a URL.")
    else:
        with st.spinner("Processing video..."):
            # Handle video input
            if uploaded_file:
                temp_video_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
                with open(temp_video_path, "wb") as f:
                    f.write(uploaded_file.read())
            elif video_url:
                temp_video_path = os.path.join(tempfile.gettempdir(), "downloaded_video.mp4")
                response = requests.get(video_url, stream=True)
                if response.status_code == 200:
                    with open(temp_video_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=1024):
                            f.write(chunk)

            # Load video
            video_clip = VideoFileClip(temp_video_path)

            # Extract audio
            audio_path = os.path.join(tempfile.gettempdir(), "temp_audio.wav")
            video_clip.audio.write_audiofile(audio_path)

            # Transcribe audio using Whisper
            model = whisper.load_model("base")
            result = model.transcribe(audio_path, word_timestamps=True)

            # Identify filler words
            detected_filler_intervals = []
            for segment in result["segments"]:
                if "words" in segment:
                    for word in segment["words"]:
                        if word["word"].strip().lower() in filler_words:
                            detected_filler_intervals.append((word["start"], word["end"]))

            # Process audio to remove fillers
            if detected_filler_intervals:
                modified_audio_path = os.path.join(tempfile.gettempdir(), "modified_audio.wav")
                audio = AudioSegment.from_wav(audio_path)
                
                for start, end in detected_filler_intervals:
                    start_ms, end_ms = int(start * 1000), int(end * 1000)
                    silence = AudioSegment.silent(duration=(end_ms - start_ms))
                    audio = audio[:start_ms] + silence + audio[end_ms:]
                
                audio.export(modified_audio_path, format="wav")
                
                # Replace original audio with modified audio in the video
                output_video_path = os.path.join(tempfile.gettempdir(), "output_video.mp4")
                video_clip.audio = VideoFileClip(modified_audio_path).audio
                video_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
                
                st.success("Processing complete! Download your video below.")
                st.video(output_video_path)
                st.download_button("Download Processed Video", open(output_video_path, "rb"), file_name="processed_video.mp4")
            else:
                st.warning("No filler words detected in the transcription.")
