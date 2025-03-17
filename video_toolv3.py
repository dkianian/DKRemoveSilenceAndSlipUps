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
import pytube
from pytube import YouTube
from urllib.parse import urlparse, parse_qs

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

whisper_model = whisper.load_model("medium")

def streamlit_ui():
    st.title("Don Kianian's AI Video Editing Tool")
    st.markdown("This tool allows you to easily and automatically remove user-selected words and silence from a video using OpenAI Whisper.")
    
    # File uploader for video
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])
    
    # Text input for video URL
#    video_url = st.text_input("Or enter a publicly-availble video URL (leave blank if uploading a file)")
    
    # Text input for filler words (comma-separated)
    filler_words_input = st.text_input("Enter filler words to remove (comma-separated)", "")
    filler_words = [word.strip().lower() for word in filler_words_input.split(",") if word.strip()]

    # Initialize filler word count
    if "filler_words_count" not in st.session_state:
        st.session_state.filler_words_count = 0

    # Persistent Buttons with Session State
    if "processing" not in st.session_state:
        st.session_state.processing = False
        st.session_state.start_time = None
        st.session_state.current_step = ""
        st.session_state.final_video_path = None
        st.session_state.input_srt_path = None
        st.session_state.output_srt_path = None
        st.session_state.processing_done = False # Flag to indicate processing is done

    # Process Video Button (Only show if processing is not done)
    if not st.session_state.processing_done:
        if st.button("Process Video", disabled=st.session_state.processing):
            st.session_state.processing = True
            st.session_state.start_time = time.time()
            st.session_state.current_step = "Starting processing..."
            # st.rerun()  # Refresh UI
            disabled = True
        
    # If processing state is set, run the main function
        if st.session_state.processing and not st.session_state.processing_done:
            try:
                main(uploaded_file, filler_words_input, video_url="")
                st.session_state.processing_done = True  # Mark as done
            except Exception as e:
                st.error(f"Error processing video: {e}")
            finally:
                st.session_state.processing = False  # Reset processing flag
                #st.rerun()  # Refresh UI to update button states

    # Display processing status & disable Process Video button when running
    if st.session_state.processing:
        st.write(f"**Current Step:** {st.session_state.current_step}")
        if st.session_state.start_time:
            elapsed_time = time.time() - st.session_state.start_time
            st.write(f"**Time Elapsed:** {timedelta(seconds=int(elapsed_time))}")

    # Step 11: Generate DL Buttons
    if st.session_state.final_video_path and os.path.exists(st.session_state.final_video_path):
        with open(st.session_state.final_video_path, "rb") as file:
            st.download_button(label="Download Trimmed Video", data=file, file_name="output_trimmed.mp4", mime="video/mp4")
    if st.session_state.input_srt_path and os.path.exists(st.session_state.input_srt_path):
        with open(st.session_state.input_srt_path, "r") as file:
            st.download_button(label="Download Original Transcript (SRT)", data=file, file_name="input_transcript.srt", mime="text/plain")
    if st.session_state.output_srt_path and os.path.exists(st.session_state.output_srt_path):
        with open(st.session_state.output_srt_path, "r") as file:
            st.download_button(label="Download Processed Transcript (SRT)", data=file, file_name="output_transcript.srt", mime="text/plain")

    # Mark processing as complete when a processed file exists
    if st.session_state.final_video_path and os.path.exists(st.session_state.final_video_path):
        st.session_state.processing_done = True  # Update flag

    # Reset Button (Only show after processing is done)
    if st.session_state.processing_done:
        st.session_state.processing = False
        st.session_state.start_time = None
        st.session_state.current_step = ""
        if st.button("Process Another Video"):
            st.session_state.processing = False
            st.session_state.processing_done = False
            st.session_state.start_time = None
            st.session_state.current_step = ""
            st.session_state.final_video_path = None
            st.session_state.input_srt_path = None
            st.session_state.output_srt_path = None
            st.rerun()  # Refresh UI

def main(uploaded_file, filler_words_input, video_url=""):
    # Check if a file path argument was passed from Streamlit
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())
            input_video = temp_file.name
#    elif video_url:
#        input_video = "downloaded_video.mp4"
#        with st.spinner("Downloading video from URL. This may take a while..."):
#            download_video_from_url(video_url, input_video)
    else:
        st.warning("Please upload a video file or enter a URL.")
        st.stop()

    # Validate the video file
    if not is_video_file_valid(input_video):
        st.error("The downloaded file is not a valid video. Please check the URL and try again.")
        st.stop()

    output_video = "output_trimmed.mp4"

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

    # Step 3: Extract audio from the unmodified video
    st.session_state.current_step = "Extracting audio..."
    with st.spinner("Extracting audio..."):
        audio_path = extract_audio(video_clip)
    progress_bar.progress(10)  # 10% progress
    print("Audio extracted successfully.")

    # Step 4: Transcribe the audio to detect filler words
    st.session_state.current_step = "Transcribing audio..."
    with st.spinner("Transcribing audio..."):
        words = transcribe_audio(audio_path, whisper_model)
    progress_bar.progress(20)  # 20% progress
    print("Audio transcribed successfully.")

    # Step 5: Generate SRT transcript for the input video
    st.session_state.current_step = "Generating SRT for unmodified video..."
    input_srt_path = "input_transcript.srt"
    generate_srt(words, input_srt_path)
    log_debug(f"DEBUG: Input video transcript saved to: {input_srt_path}")
    progress_bar.progress(25)  # 25% progress
    print("Input video SRT generated successfully.")

# Step 6: Process filler words if provided
    modified_audio_path = audio_path  # Default to original audio
    if filler_words:
        # Step 6a: Detect filler words
        st.session_state.current_step = "Detecting filler words..."
        with st.spinner("Detecting filler words..."):
            st.session_state.filler_words_count, filler_intervals = detect_filler_words(words, filler_words)
            filler_intervals = merge_intervals(filler_intervals)
        progress_bar.progress(30)
        log_debug(f"DEBUG: Filler words detected")
        print("Identifying filler words...")

        # Step 6b: Replace filler words with silence
        st.session_state.current_step = "Removing filler words..."
        with st.spinner("Removing filler words..."):
            modified_audio_path = "temp_modified_audio.wav"
            replace_filler_words_with_silence(audio_path, filler_intervals, modified_audio_path, buffer=0.015)
            progress_bar.progress(50)
            log_debug("Filler words removed.")
            print("Removed filler words...")
    else:
        log_debug("No filler words provided. Proceeding with original audio for silence removal.")
        st.session_state.filler_words_count = 0

    # Step 7: Detect non-silent intervals (always runs, using modified or original audio)
    st.session_state.current_step = "Detecting silence..."
    with st.spinner("Detecting silence..."):
        non_silent_times = detect_non_silent_intervals(modified_audio_path)
    progress_bar.progress(60)  # Adjusted progress
        
    # Step 8: Trim the video and audio to remove silent segments
    st.session_state.current_step = "Trimming video..."
    with st.spinner("Trimming video..."):
        final_video_path = trim_video_and_audio(video_clip, non_silent_times, output_video)
    progress_bar.progress(75)
    print("Trimmed video to remove silences...")
    log_debug(f"Silences removed. Final video saved to: {final_video_path}")

    # Step 9: Generate SRT for the output video
    st.session_state.current_step = "Generating output video SRT..."
    with st.spinner("Generating output video SRT..."):
        output_audio_path = "temp_output_audio.wav"
        extract_audio(load_video(final_video_path), output_audio_path)
        output_words = transcribe_audio(output_audio_path, whisper_model)
        output_srt_path = "output_transcript.srt"
        generate_srt(output_words, output_srt_path)
        log_debug(f"DEBUG: Output video transcript saved to: {output_srt_path}")
    progress_bar.progress(80)
    print("Generated output SRT...")

    # Clean up temporary files
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
    if "final_clip" in locals():
        final_clip.close()  # Close the trimmed video if it was created

    write_progress(100)
    # Final UI Updates After Processing Completes
    progress_bar.progress(100)  # Mark progress as complete
    st.success(f"Processing complete! Download the final video and transcripts below. {st.session_state.filler_words_count} filler words removed.")

    # Store the file paths in the session state
    st.session_state.final_video_path = final_video_path
    st.session_state.input_srt_path = input_srt_path
    st.session_state.output_srt_path = output_srt_path

    # Reset Processing State
    st.session_state.processing = False
    st.session_state.current_step = ""

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

def is_video_file_valid(file_path):
    """Check if the video file is valid using FFmpeg."""
    try:
        # Use FFmpeg to probe the file
        command = [ffmpeg_path, "-v", "error", "-i", file_path, "-f", "null", "-"]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Error validating video file: {e}")
        return False

def load_video(video_path):
    """Load a video file using moviepy."""
    return VideoFileClip(video_path)

def download_video_from_url(url, output_path):
    """Download a video from a URL (Google Drive, YouTube, or direct link)."""
    try:
        # Handle Google Drive links
        if "drive.google.com" in url:
            # Extract file ID from the URL
            if "/file/d/" in url:
                # Example: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
                file_id = url.split("/file/d/")[1].split("/")[0]
            elif "id=" in url:
                # Example: https://drive.google.com/uc?export=download&id=FILE_ID
                file_id = parse_qs(urlparse(url).query).get('id', [None])[0]
            else:
                raise Exception("Invalid Google Drive URL. Could not extract file ID.")

            if not file_id:
                raise Exception("Invalid Google Drive URL. Could not extract file ID.")

            # Construct the direct download URL
            direct_download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            # Use a session to handle cookies and redirects
            session = requests.Session()
            response = session.get(direct_download_url, stream=True)
            print(f"Final download URL: {response.url}")

            # Check if Google Drive is serving a confirmation page
            if "confirm=" in response.url:
                # Extract the confirmation token
                confirm_token = re.search(r"confirm=([^&]+)", response.url).group(1)
                # Construct the confirmed download URL
                confirmed_download_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={confirm_token}"
                response = session.get(confirmed_download_url, stream=True)

            if response.status_code == 200:
                with open(output_path, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:  # Filter out keep-alive chunks
                            file.write(chunk)
                print(f"Google Drive video downloaded and saved to: {os.path.abspath(output_path)}")

                # Log the file size
                file_size = os.path.getsize(output_path)
                print(f"Downloaded file size: {file_size} bytes")

                # Validate the downloaded file
                if not is_video_file_valid(output_path):
                    os.remove(output_path)  # Delete the invalid file
                    raise Exception("Downloaded file is not a valid video.")
            else:
                raise Exception(f"Failed to download Google Drive video. Status code: {response.status_code}")

        # Handle YouTube links
        elif "youtube.com" in url or "youtu.be" in url:
            yt = YouTube(url)
            stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
            if not stream:
                raise Exception("No suitable video stream found on YouTube.")
            stream.download(filename=output_path)
            print(f"YouTube video downloaded and saved to: {output_path}")

            # Validate the downloaded file
            if not is_video_file_valid(output_path):
                os.remove(output_path)  # Delete the invalid file
                raise Exception("Downloaded file is not a valid video.")

        # Handle direct links (e.g., MP4 files)
        else:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(output_path, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
                print(f"Video downloaded and saved to: {output_path}")
            
            # Validate the downloaded file
                if not is_video_file_valid(output_path):
                    os.remove(output_path)  # Delete the invalid file
                    raise Exception("Downloaded file is not a valid video.")
            else:
                raise Exception(f"Failed to download video from URL. Status code: {response.status_code}")

    except Exception as e:
        raise Exception(f"Error downloading video: {e}")
    
    print(f"Downloaded file size: {os.path.getsize(output_path)} bytes")

def extract_audio(video_clip, audio_path="temp_audio.wav"):
    """Extract audio from a video clip and save it as a WAV file."""
    video_clip.audio.write_audiofile(audio_path)
    return audio_path

def detect_non_silent_intervals(audio_path, top_db=30):
    """Detect non-silent intervals in the audio file using librosa."""
    y, sr = librosa.load(audio_path, sr=None)
    y = librosa.util.normalize(y)
    non_silent_intervals = librosa.effects.split(y, top_db=top_db)
    non_silent_times = [(start / sr, end / sr) for start, end in non_silent_intervals
        if (end - start) / sr > 0.5] # e.g. 0.5 seconds minumum duration
    log_debug(f"Detected non-silent intervals: {non_silent_times}")
    log_debug(f"DEBUG: Total non-silent segments: {len(non_silent_times)}")
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

def detect_filler_words(words, filler_words):
    """Detect filler words in the transcription."""
    filler_intervals = []
    filler_words_count = 0
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
                filler_words_count += 1
                detected_any = True
                log_debug(f"Detected filler word: {word_text} ({start_time}s - {end_time}s)")
    if not detected_any:
        log_debug("WARNING: No filler words detected in transcription. Check Whisper output.")
    log_debug(f"DEBUG: Detected filler intervals: {filler_intervals}")
    return filler_words_count, filler_intervals

def replace_filler_words_with_silence(audio_path, filler_intervals, output_audio_path, buffer=0.01):
    """
    Replace filler word segments in the audio with silence, with a small buffer added to the intervals.
    
    Parameters:
    - audio_path: Path to the input audio file.
    - filler_intervals: List of (start, end) tuples representing filler word intervals.
    - output_audio_path: Path to save the modified audio file.
    - buffer: Time in seconds to extend the start and end of each interval (default: 0.05 seconds).
    """
    # Load the audio file
    audio = AudioSegment.from_wav(audio_path)
    
    # Iterate over filler intervals and replace them with silence
    for start, end in sorted(filler_intervals, reverse=True):
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

# def trim_video_and_audio(video_clip, non_silent_times, output_video_path):
#    """Trim the video and audio to match the detected non-silent intervals."""
#    video_subclips = [video_clip.subclip(start, end) for (start, end) in non_silent_times]
#    final_clip = concatenate_videoclips(video_subclips)
#    final_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
#    return output_video_path

def trim_video_and_audio(video_clip, non_silent_times, output_video_path):
    if not non_silent_times:
        log_debug("WARNING: No non-silent intervals detected. Copying original video.")
        video_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
        return output_video_path
    
    log_debug(f"Trimming with intervals: {non_silent_times}")
    video_subclips = []
    for start, end in non_silent_times:
        if start < end and end <= video_clip.duration:
            video_subclips.append(video_clip.subclip(start, end))
            log_debug(f"Added subclip: {start}s to {end}s")
        else:
            log_debug(f"Skipping invalid interval: {start}s to {end}s")
    
    if not video_subclips:
        log_debug("ERROR: No valid subclips to concatenate. Using original video.")
        video_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
        return output_video_path
    
    final_clip = concatenate_videoclips(video_subclips)
    log_debug(f"Concatenated {len(video_subclips)} subclips")
    final_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
    log_debug(f"Final video written to {output_video_path}")
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

if __name__ == "__main__":
    streamlit_ui()