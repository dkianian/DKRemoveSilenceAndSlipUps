import os
# ✅ Fix PyTorch watcher error in Streamlit
os.environ["TORCH_USE_RTLD_GLOBAL"] = "1"
# ✅ Ensure Streamlit does NOT track file changes
os.environ["STREAMLIT_DISABLE_FILE_WATCHER"] = "true"
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, concatenate_audioclips
from pydub import AudioSegment
from pydub.silence import split_on_silence
import librosa
import numpy as np
import soundfile as sf
import json
import re
import whisper
import requests
import sys
import streamlit as st
import tempfile
import shutil
import subprocess
import imageio_ffmpeg
import time 
from datetime import timedelta
import threading

# Check if the script is running in Streamlit
RUNNING_IN_STREAMLIT = "streamlit" in sys.argv[0]

# Ensure FFmpeg is available
ffmpeg_path = shutil.which("ffmpeg")
  
from pydub import AudioSegment

# Progress file to communicate with Streamlit
PROGRESS_FILE = "progress.txt"
if shutil.which("ffmpeg") is None:
    print("FFmpeg not found. Please install FFmpeg and add it to your PATH.")
def write_progress(progress):
    """Write progress to a file."""
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(progress))
        
# Debug log file
DEBUG_LOG_FILE = "debug_log.txt"

whisper_model = whisper.load_model("tiny.en")

def streamlit_ui():
    st.title("Don Kianian's Automated Video Editor")
    st.markdown("This tool allows you to process long-form video by removing user-selected words and silence.")
    
    # File uploader for video
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])
    
    # Text input for video URL
    video_url = st.text_input("Or enter a video URL (leave blank if uploading a file)")
    
    # Text input for filler words (comma-separated)
    filler_words_input = st.text_input("Enter filler words to remove (comma-separated)", "")
    filler_words = [word.strip().lower() for word in filler_words_input.split(",") if word.strip()]

    # Persistent Buttons with Session State
    if "processing" not in st.session_state:
        st.session_state.processing = False
        st.session_state.start_time = None
        st.session_state.current_step = ""
        st.session_state.transcription_result = None # Initialize transcription result

    col1, col2 = st.columns(2)  # Arrange buttons side by side

    # Process Video Button
    with col1:
        if st.button("Process Video", disabled=st.session_state.processing):
            st.session_state.processing = True
            st.session_state.start_time = time.time()
            st.session_state.current_step = "Starting processing..."
            main(uploaded_file, video_url, filler_words_input)

    # Reset Button
    with col2:
        if st.button("Reset"):
            st.session_state.processing = False
            st.session_state.start_time = None
            st.session_state.current_step = ""
            st.session_state.transcription_result = None # Reset transcription result
            st.experimental_rerun()

    # Display processing status if running
    if st.session_state.processing:
        st.write(f"**Current Step:** {st.session_state.current_step}")
        if st.session_state.start_time:
            elapsed_time = time.time() - st.session_state.start_time
            st.write(f"**Time Elapsed:** {timedelta(seconds=int(elapsed_time))}")

def log_debug(message):
    """Log a debug message to a file."""
    if not os.path.exists(DEBUG_LOG_FILE):
        with open(DEBUG_LOG_FILE, "w") as log_file:
            pass  # Create an empty file
    with open(DEBUG_LOG_FILE, "a") as log_file:
        log_file.write(message + "\n")

def consolidate_debug_log(input_file, output_file):
    """Consolidate and simplify the debug log to focus on critical information."""
    if not os.path.exists(input_file):
        print(f"Input file '{input_file}' does not exist.")
        return
    keywords = [
        "Filler words to filter out:",
        "Detected filler word:",
        "DEBUG - Transcribed Word:",
        "DEBUG: Filler intervals to remove:",
        "DEBUG: Filler intervals before merging:",
        "DEBUG: Merged filler intervals:",
        "DEBUG: Adding segment:",
        "DEBUG: Skipping filler word:",
        "DEBUG: Adding final segment",
        "DEBUG: Concatenating clips",
        "DEBUG: Writing final video",
        "DEBUG: Detected filler word",
        "DEBUG: Merged filler intervals",
        "DEBUG: Final video duration",
        "DEBUG: Final audio duration",
        "DEBUG: Concatenating clips",
        "DEBUG: Number of video segments to concatenate:",
        "DEBUG: Number of audio segments to concatenate:",
        "ERROR",
        "WARNING",
        "Skipping segment",
        "Filler words removed",
        "No filler words detected. Skipping removal.",
        "Filler words removed. Final video saved to:",
        "DEBUG: Concatenating video and audio segments...",
        "DEBUG: Skipping filler word from",
    ]
    with open(input_file, "r") as infile:
        log_lines = infile.readlines()
    consolidated_lines = [line for line in log_lines if any(keyword in line for keyword in keywords)]
    with open(output_file, "w") as outfile:
        outfile.writelines(consolidated_lines)
    print(f"Consolidated debug log saved to: {output_file}")

def load_video(video_path):
    """Load a video file using moviepy."""
    return VideoFileClip(video_path)

def download_video_from_url(url, output_path):
    """Download a video from a URL and save it to the specified path."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"Video downloaded and saved to: {output_path}")
    else:
        raise Exception(f"Failed to download video from URL. Status code: {response.status_code}")

def extract_audio(video_clip, audio_path="temp_audio.wav"):
    """Extract audio from a video clip and save it as a WAV file."""
    video_clip.audio.write_audiofile(audio_path)
    return audio_path

def detect_non_silent_intervals(audio_path, top_db=40):
    """Detect non-silent intervals in the audio file using librosa."""
    y, sr = librosa.load(audio_path, sr=None)
    non_silent_intervals = librosa.effects.split(y, top_db=top_db)
    non_silent_times = [(start / sr, end / sr) for start, end in non_silent_intervals]
    return non_silent_times

def transcribe_audio(audio_path, model):
    """Transcribe the audio using OpenAI Whisper."""
    result = model.transcribe(audio_path, word_timestamps=True)
    if not result or "segments" not in result or result["segments"] is None:
        st.error("Error: Whisper failed to generate transcription segments.")
        log_debug("ERROR: No segments found in transcription. Possible empty audio or model failure.")
        return []

    words_only = [
        {"word": word["word"], "start": word["start"], "end": word["end"]}
        for segment in result["segments"] if "words" in segment and segment["words"] is not None
        for word in segment["words"]
    ]

    for word_data in words_only:
        log_debug(f"DEBUG - Transcribed Word: {word_data['word']} ({word_data['start']}s - {word_data['end']}s)")
    log_debug("DEBUG: Transcription Output:")
    for segment in result["segments"]:
        log_debug(json.dumps(segment, indent=2))
    return result["segments"]
    """Run transcription in a background thread."""
    try:
        st.session_state.transcription_result = model.transcribe(audio_path, word_timestamps=True)
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        st.session_state.transcription_result = None

def detect_filler_words(words, filler_words):
    """Detect filler words in the transcription."""
    filler_intervals = []
    filler_words = [word.strip().lower() for word in filler_words]
    log_debug(f"DEBUG: Filler words to detect: {filler_words}")
    detected_any = False
    for segment in words:
        if not segment.get("words"):
            log_debug(f"Skipping empty segment: {segment}")
            continue
        for word in segment["words"]:
            word_text = word["word"].strip().lower()
            log_debug(f"Checking word: {word_text} against filler list: {filler_words}")
            if word_text in filler_words:
                start_time = word["start"]
                end_time = word["end"]
                filler_intervals.append((start_time, end_time))
                detected_any = True
                log_debug(f"Detected filler word: {word_text} ({start_time}s - {end_time}s)")
    if not detected_any:
        log_debug("WARNING: No filler words detected in transcription. Check Whisper output.")
    log_debug(f"DEBUG: Detected filler intervals: {filler_intervals}")
    return filler_intervals

def replace_filler_words_with_silence(audio_path, filler_intervals, output_audio_path, buffer=0.01):
    """
    Replace filler word segments in the audio with silence, with a small buffer added to the intervals.
    
    Parameters:
    - audio_path: Path to the input audio file.
    - filler_intervals: List of (start, end) tuples representing filler word intervals.
    - output_audio_path: Path to save the modified audio file.
    - buffer: Time in seconds to extend the start and end of each interval (default: 0.01 seconds).
    """
    # Load the audio file
    audio = AudioSegment.from_wav(audio_path)
    
    # Iterate over filler intervals and replace them with silence
    for start, end in filler_intervals:
        # Add buffer to the start and end of the interval
        start_with_buffer = max(0, start - buffer)  # Ensure start doesn't go below 0
        end_with_buffer = min(len(audio) / 1000, end + buffer)  # Ensure end doesn't exceed audio length
        
        # Convert seconds to milliseconds
        start_ms = int(start_with_buffer * 1000)
        end_ms = int(end_with_buffer * 1000)
        
        # Generate silence for the interval (including buffer)
        silence = AudioSegment.silent(duration=end_ms - start_ms)
        
        # Replace the interval (including buffer) with silence
        audio = audio[:start_ms] + silence + audio[end_ms:]
    
    # Export the modified audio
    audio.export(output_audio_path, format="wav")
    log_debug(f"DEBUG: Filler words replaced with silence (with buffer). Modified audio saved to: {output_audio_path}")

def trim_video_and_audio(video_clip, non_silent_times, output_video_path):
    """Trim the video and audio to match the detected non-silent intervals."""
    video_subclips = [video_clip.subclip(start, end) for (start, end) in non_silent_times]
    final_clip = concatenate_videoclips(video_subclips)
    final_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
    return output_video_path

def get_filler_words(filler_words_input):
    """Process filler words input from Streamlit or CLI."""
    return [word.strip().lower() for word in filler_words_input.split(",")] if filler_words_input.strip() else []

def merge_intervals(intervals):
    """Merge overlapping or adjacent intervals."""
    if not intervals:
        return []
    intervals = [(float(start), float(end)) for start, end in intervals]
    intervals.sort(key=lambda x: x[0])
    merged = []
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end + 0.05:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))
    log_debug(f"DEBUG: Merged filler intervals after conversion: {merged}")
    return merged

def generate_srt(transcription, output_srt_path):
    """Generate an SRT file from the transcription."""
    with open(output_srt_path, "w") as srt_file:
        for i, segment in enumerate(transcription):
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"]
            
            # Convert start and end times to SRT format (HH:MM:SS,mmm)
            start_time_srt = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02},{int((start_time % 1) * 1000):03}"
            end_time_srt = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02},{int((end_time % 1) * 1000):03}"
            
            # Write the SRT entry
            srt_file.write(f"{i + 1}\n")
            srt_file.write(f"{start_time_srt} --> {end_time_srt}\n")
            srt_file.write(f"{text}\n\n")

def main(uploaded_file, video_url, filler_words_input):
    # Initialize final_clip to None
    final_clip = None
    # Check if a file path argument was passed from Streamlit
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())
            input_video = temp_file.name
    elif video_url:
        input_video = "downloaded_video.mp4"
        download_video_from_url(video_url, input_video)
    else:
        st.warning("Please upload a video file or enter a URL.")
        st.stop()
    
    output_video = "output_trimmed.mp4"

    # Log start time
    start_time = time.time()

    # Clean old debug logs, SRTs, and output
    if os.path.exists(DEBUG_LOG_FILE):
        os.remove(DEBUG_LOG_FILE)
    if os.path.exists("consolidated_debug_log.txt"):
        os.remove("consolidated_debug_log.txt")
    if os.path.exists("input_transcript.srt"):
        os.remove("input_transcript.srt")
    if os.path.exists("output_transcript.srt"):
        os.remove("output_transcript.srt")
    if os.path.exists("output_trimmed.mp4"):
        os.remove("output_trimmed.mp4")
    
    # Initialize progress
    write_progress(0)
    if "progress_bar" not in st.session_state:
        st.session_state.progress_bar = st.progress(0)  # Create a single instance
    progress_bar = st.session_state.progress_bar

    # Create placeholders for time elapsed and remaining
    time_elapsed_placeholder = st.empty()
    time_remaining_placeholder = st.empty()

    # Step 1: Get filler words to filter out
    st.session_state.current_step = "Getting filler words..."
    filler_words = get_filler_words(filler_words_input.strip()) if filler_words_input else []
    log_debug(f"Filler words to filter out: {filler_words}")

    # Step 2: Load the video
    st.session_state.current_step = "Loading video..."
    with st.spinner("Loading video..."):
        video_clip = load_video(input_video)
    progress_bar.progress(5)  # 5% progress
    print("Video loaded successfully.")

    # Update time elapsed and remaining
    elapsed_time = time.time() - start_time
    estimated_total_time = elapsed_time / 0.05  # Assuming 5% progress
    remaining_time = estimated_total_time - elapsed_time
    time_elapsed_placeholder.text(f"Time Elapsed: {timedelta(seconds=int(elapsed_time))}")
    time_remaining_placeholder.text(f"Estimated Time Remaining: {timedelta(seconds=int(remaining_time))}")

    # Step 3: Extract audio from the unmodified video
    st.session_state.current_step = "Extracting audio..."
    with st.spinner("Extracting audio..."):
        audio_path = extract_audio(video_clip)
    progress_bar.progress(10)  # 10% progress
    print("Audio extracted")

    # Update time elapsed and remaining
    elapsed_time = time.time() - start_time
    estimated_total_time = elapsed_time / 0.10  # Assuming 10% progress
    remaining_time = estimated_total_time - elapsed_time
    time_elapsed_placeholder.text(f"Time Elapsed: {timedelta(seconds=int(elapsed_time))}")
    time_remaining_placeholder.text(f"Estimated Time Remaining: {timedelta(seconds=int(remaining_time))}")

    # Step 4: Transcribe the audio to detect filler words
    st.session_state.current_step = "Transcribing audio..."
    with st.spinner("Transcribing audio..."):
        # Run transcription in a background thread
        transcription_thread = threading.Thread(target=transcribe_audio, args=(audio_path, whisper_model))
        transcription_thread.start()

        # Wait for transcription to complete
        while transcription_thread.is_alive():
            time.sleep(1)  # Check every second
        st.write("Transcription in progress...")

        transcription_thread.join()

    # Get transcription result
    transcription_result = st.session_state.get("transcription_result")
    if st.session_state.transcription_result is None:
        st.error("Transcription failed. Please check the audio file and try again.")
        st.stop()

    words = transcription_result["segments"]
    progress_bar.progress(30)  # 30% progress
    print("Audio transcribed successfully.")

    # Update time elapsed and remaining
    elapsed_time = time.time() - start_time
    estimated_total_time = elapsed_time / 0.3  # Assuming 30% progress
    remaining_time = estimated_total_time - elapsed_time
    time_elapsed_placeholder.text(f"Time Elapsed: {timedelta(seconds=int(elapsed_time))}")
    time_remaining_placeholder.text(f"Estimated Time Remaining: {timedelta(seconds=int(remaining_time))}")

    # Step 5: Generate SRT transcript for the input video
    st.session_state.current_step = "Generating SRT transcript..."
    input_srt_path = "input_transcript.srt"
    generate_srt(words, input_srt_path)
    log_debug(f"DEBUG: Input video transcript saved to: {input_srt_path}")
    progress_bar.progress(35)  # 35% progress
    print("Input video SRT generated successfully.")

    # Update time elapsed and remaining
    elapsed_time = time.time() - start_time
    estimated_total_time = elapsed_time / 0.35  # Assuming 35% progress
    remaining_time = estimated_total_time - elapsed_time
    time_elapsed_placeholder.text(f"Time Elapsed: {timedelta(seconds=int(elapsed_time))}")
    time_remaining_placeholder.text(f"Estimated Time Remaining: {timedelta(seconds=int(remaining_time))}")

    # Step 6: Detect filler words in the unmodified video
    st.session_state.current_step = "Detecting filler words..."
    if filler_words:
        with st.spinner("Detecting filler words..."):
            filler_intervals = detect_filler_words(words, filler_words)
            filler_intervals = merge_intervals(filler_intervals)
        progress_bar.progress(40)  # 40% progress
        print("Identifying filler words...")

        # Update time elapsed and remaining
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time / 0.4  # Assuming 40% progress
        remaining_time = estimated_total_time - elapsed_time
        time_elapsed_placeholder.text(f"Time Elapsed: {timedelta(seconds=int(elapsed_time))}")
        time_remaining_placeholder.text(f"Estimated Time Remaining: {timedelta(seconds=int(remaining_time))}")

        # Step 7: Replace filler words with silence in the audio (with buffer)
        st.session_state.current_step = "Removing filler words..."
        with st.spinner("Removing filler words..."):
            modified_audio_path = "temp_modified_audio.wav"
            replace_filler_words_with_silence(audio_path, filler_intervals, modified_audio_path, buffer=0.01)
        progress_bar.progress(50)  # 50% progress
        print("Replacing filler words...")

        # Update time elapsed and remaining
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time / 0.5  # Assuming 50% progress
        remaining_time = estimated_total_time - elapsed_time
        time_elapsed_placeholder.text(f"Time Elapsed: {timedelta(seconds=int(elapsed_time))}")
        time_remaining_placeholder.text(f"Estimated Time Remaining: {timedelta(seconds=int(remaining_time))}")

        # Step 8: Detect non-silent intervals in the modified audio
        non_silent_times = detect_non_silent_intervals(modified_audio_path)
        
        # Step 9: Trim the video and audio to remove the silent segments
        st.session_state.current_step = "Trimming video and audio..."
        with st.spinner("Trimming video and audio..."):
            if filler_words:
                final_video_path = trim_video_and_audio(video_clip, non_silent_times, output_video)
                final_clip = load_video(final_video_path) # Load the final clip
                progress_bar.progress(60)  # 60% progress
                print("Removing filler words...")
                log_debug(f"Filler words removed. Final video saved to: {final_video_path}")
            else:
                log_debug("No filler words detected. Skipping removal.")
                final_video_path = input_video

        # Update time elapsed and remaining
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time / 0.6  # Assuming 60% progress
        remaining_time = estimated_total_time - elapsed_time
        time_elapsed_placeholder.text(f"Time Elapsed: {timedelta(seconds=int(elapsed_time))}")
        time_remaining_placeholder.text(f"Estimated Time Remaining: {timedelta(seconds=int(remaining_time))}")

    else:
        log_debug("No filler words detected. Skipping removal.")
        final_video_path = input_video  # Skip filler word removal if no words provided

    # Step 10: Generate SRT transcript for the output trimmed video
    st.session_state.current_step = "Generating SRT transcript..."
    output_audio_path = "temp_output_audio.wav"
    extract_audio(load_video(final_video_path), output_audio_path)
    output_words = transcribe_audio(output_audio_path, model=whisper_model)
    output_srt_path = "output_transcript.srt"
    generate_srt(output_words, output_srt_path)
    log_debug(f"DEBUG: Output video transcript saved to: {output_srt_path}")
    progress_bar.progress(80)  # 80% progress
    print("Generating final video and output SRT...")

    # Update time elapsed and remaining
    elapsed_time = time.time() - start_time
    estimated_total_time = elapsed_time / 0.8  # Assuming 80% progress
    remaining_time = estimated_total_time - elapsed_time
    time_elapsed_placeholder.text(f"Time Elapsed: {timedelta(seconds=int(elapsed_time))}")
    time_remaining_placeholder.text(f"Estimated Time Remaining: {timedelta(seconds=int(remaining_time))}")

    # Clean up temporary files
    st.session_state.current_step = "Tidying up..."
    os.remove(audio_path)
    if os.path.exists("temp_modified_audio.wav"):
        os.remove("temp_modified_audio.wav")
    if os.path.exists("temp_output_audio.wav"):
        os.remove("temp_output_audio.wav")
    progress_bar.progress(90)  # 90% progress
    print("Cleaning up temp files...")

    print(f"Trimmed video saved to: {final_video_path}")
    print(f"Debug log saved to: {DEBUG_LOG_FILE}")
    consolidate_debug_log(DEBUG_LOG_FILE, "consolidated_debug_log.txt")
    
    video_clip.close()  # Close the video file
    # Close the trimmed video if it was created
    if "final_clip" is not None:
        final_clip.close()

    st.success("Processing completed successfully!")
    write_progress(100)
    # Final UI Updates After Processing Completes
    progress_bar.progress(100)  # Mark progress as complete
    st.success("Processing complete! Download the final video and transcripts below.")

    # Step 11: Generate DL Buttons
    if os.path.exists("output_trimmed.mp4"):
        with open("output_trimmed.mp4", "rb") as file:
            st.download_button(label="Download Trimmed Video", data=file, file_name="output_trimmed.mp4", mime="video/mp4")
    if os.path.exists("input_transcript.srt"):
        with open("input_transcript.srt", "r") as file:
            st.download_button(label="Download Original Transcript (SRT)", data=file, file_name="input_transcript.srt", mime="text/plain")
    if os.path.exists("output_transcript.srt"):
        with open("output_transcript.srt", "r") as file:
            st.download_button(label="Download Processed Transcript (SRT)", data=file, file_name="output_transcript.srt", mime="text/plain")

    # Reset Processing State
    st.session_state.processing = False

if __name__ == "__main__":
    streamlit_ui()