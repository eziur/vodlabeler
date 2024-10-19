import warnings
import subprocess
import sys
import os
import re
from whisper import load_model
from faster_whisper import WhisperModel
from datetime import timedelta
import numpy as np
import io

# Force UTF-8 encoding
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
my_env = os.environ.copy()
my_env["PYTHONIOENCODING"] = "utf-8"
my_env["PYTHONUTF8"] = "1"


def extract_audio_ffmpeg(video_path, start_time, duration):
    """
    Use FFmpeg to extract audio from the video starting at start_time for duration seconds.
    """
    print(f"Extracting audio from {start_time}s to {start_time + duration}s of the video: {video_path}")

    # FFmpeg command to extract audio from the video
    command = [
        'ffmpeg',
        '-ss', str(start_time),
        '-i', video_path,
        '-t', str(duration),
        '-f', 'wav',
        '-vn',  # Convert to WAV format, no video
        'pipe:1'  # Pipe output to stdout
    ]

    # Run the FFmpeg command and capture stdout (the audio data)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    audio_data, stderr = process.communicate()

    if process.returncode != 0:
        print(f"Error during audio extraction:\n{stderr.decode('utf-8')}")
        return None

    print("Audio extraction successful.")
    return audio_data


def transcribe_audio(audio_data, start_time):
    print("Loading Faster Whisper model for transcription.")

    # Load the tiny model for speed; you can adjust the model size ('tiny', 'base', etc.)
    model = WhisperModel("tiny", device="cpu", compute_type="float32")  # Explicitly setting to float32

    print("Starting transcription from audio data.")

    # Convert audio data to a byte stream for Faster Whisper to process
    audio_stream = io.BytesIO(audio_data)

    # Perform transcription
    segment_generator, _ = model.transcribe(audio_stream, beam_size=5)

    # Convert the generator to a list so we can work with it
    segments = list(segment_generator)

    # Create a list of adjusted segments
    adjusted_segments = []

    for segment in segments:
        adjusted_segment = {
            'start': segment.start + start_time,
            'end': segment.end + start_time,
            'text': segment.text
        }
        adjusted_segments.append(adjusted_segment)

    # Check if any segments were generated
    if not adjusted_segments:
        print("Warning: No segments were generated during transcription.")

    print(f"Transcription completed. Total segments: {len(adjusted_segments)}")

    return adjusted_segments


def run_llama(prompt):
    llama_command = ["ollama", "run", "llama3.1"]

    process = subprocess.Popen(
        llama_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=my_env
    )

    stdout_data, stderr_data = process.communicate(input=prompt)

    if process.returncode != 0:
        print(f"Error running LLaMA (return code {process.returncode}):\n{stderr_data}")
        return ''

    output = stdout_data.strip() if stdout_data.strip() else stderr_data.strip()
    return output


def parse_llama_output(output):
    chapters = []
    # This pattern specifically matches the format: [interval_start, interval_end] Label/Description
    pattern = re.compile(r'\[(\d+:\d{2}:\d{2}),\s*(\d+:\d{2}:\d{2})\]\s*(.*)')

    for line in output.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        match = pattern.match(line)
        if match:
            start_str, end_str, label = match.groups()

            # Convert time (hh:mm:ss) to seconds
            def hms_to_seconds(hms_str):
                parts = hms_str.strip().split(':')
                parts = [int(p) for p in parts]
                while len(parts) < 3:
                    parts.insert(0, 0)
                hours, minutes, seconds = parts
                return hours * 3600 + minutes * 60 + seconds

            start_sec = hms_to_seconds(start_str)
            end_sec = hms_to_seconds(end_str)

            chapters.append({
                "start": start_sec,
                "end": end_sec,
                "label": label.strip()
            })

    return chapters


def add_chapters_to_video(video_path, chapters):
    print(f"Extracting metadata from: {video_path}")

    extract_metadata_cmd = ['ffmpeg', '-y', '-i', video_path, '-f', 'ffmetadata', 'metadata.txt']
    extract_process = subprocess.Popen(extract_metadata_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                       env=my_env)
    stdout, stderr = extract_process.communicate()

    if extract_process.returncode != 0:
        print(f"Error extracting metadata:\n{stderr}")
        return

    print("Building chapter metadata...")
    with open("metadata.txt", "a", encoding='utf-8') as f:
        for chapter in chapters:
            f.write(
                f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={int(chapter['start'] * 1000)}\nEND={int(chapter['end'] * 1000)}\ntitle={chapter['label']}\n")

    print("Adding metadata back into the video...")

    base, ext = os.path.splitext(video_path)
    output_video_path = f"{base}_with_chapters{ext}"

    add_metadata_cmd = ['ffmpeg', '-y', '-i', video_path, '-i', 'metadata.txt', '-map_metadata', '1', '-codec', 'copy',
                        output_video_path]

    add_process = subprocess.Popen(add_metadata_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   env=my_env)
    stdout, stderr = add_process.communicate()

    if add_process.returncode != 0:
        print(f"Error adding metadata:\n{stderr}")
        return

    print(f"Chapters successfully added to: {output_video_path}")


def label_intervals_with_ollama(segments, start_time, chunk_duration):
    print(f"Labeling 50 segments with LLaMA...")

    # Calculate interval length to get approximately 50 segments
    interval_length = chunk_duration / 50

    # Calculate number of intervals
    num_intervals = 50

    intervals = []

    for i in range(num_intervals):
        interval_start = start_time + i * interval_length
        interval_end = min(start_time + (i + 1) * interval_length, start_time + chunk_duration)

        # Collect segments in this interval
        interval_segments = [segment for segment in segments if
                             segment['start'] < interval_end and segment['end'] > interval_start]

        # Build transcript text
        transcript_lines = []
        for segment in interval_segments:
            start_time_str = str(timedelta(seconds=int(segment['start'])))
            end_time_str = str(timedelta(seconds=int(segment['end'])))
            text = segment['text'].strip()
            transcript_lines.append(f"[{start_time_str} --> {end_time_str}] {text}")
        interval_transcript = "\n".join(transcript_lines)

        if not interval_transcript.strip():
            print(f"No transcription data for interval {i}. Skipping...")
            continue

        # Build prompt
        prompt = f"""
Your output must be strictly in the following format:
[interval_start, interval_end] Label/Description

You are an expert analyzing a gaming VOD. Provide one brief label for these segments of the video, spanning from {str(timedelta(seconds=int(interval_start)))} to {str(timedelta(seconds=int(interval_end)))}.

Here it is:
{interval_transcript}

Your output MUST be strictly in the following format:
[interval_start, interval_end] Label/Description
"""

        # Initialize variables for retry loop
        attempt = 0
        max_attempts = 3
        labeled_segment = None

        while attempt < max_attempts and not labeled_segment:
            attempt += 1
            print(f"Attempt {attempt} for interval {i}")
            # Run LLaMA
            output = run_llama(prompt)
            if output:
                print(f"LLaMA raw output for interval {i}:\n{output}")
                parsed_output = parse_llama_output(output)
                if parsed_output and len(parsed_output) == 1:
                    # Check if timestamps match the interval range
                    parsed_start = parsed_output[0]['start']
                    parsed_end = parsed_output[0]['end']
                    if parsed_start == int(interval_start) and parsed_end == int(interval_end):
                        labeled_segment = parsed_output[0]
                        intervals.append(labeled_segment)
                    else:
                        print(
                            f"Invalid output: Timestamps do not match the interval range for interval {i}. Retrying...")
                else:
                    print(f"Invalid output from LLaMA for interval {i}. Expected exactly one label. Retrying...")
            else:
                print(f"No output from LLaMA for interval {i}. Retrying...")

        if not labeled_segment:
            print(f"Failed to get valid output for interval {i} after {max_attempts} attempts.")

    print("LLaMA labeling of intervals completed successfully.")
    print(intervals)
    return intervals


def group_labeled_intervals_with_llama(labeled_intervals):
    print("Grouping labeled intervals into larger chapters with LLaMA...")

    # Build a text representation of the labeled intervals
    intervals_text = ""
    for interval in labeled_intervals:
        start_str = str(timedelta(seconds=int(interval['start'])))
        end_str = str(timedelta(seconds=int(interval['end'])))
        label = interval['label']
        intervals_text += f"[{start_str} --> {end_str}] {label}\n"

    # Build the prompt
    prompt = f"""
Your output MUST be strictly in the following format:
[interval_start, interval_end] Concise label

You are an expert analyzing segments of a gaming VOD. Group these labeled video segments into fewer, larger segments. Here they are:

{intervals_text}

Your output MUST be strictly in the following format:
[interval_start, interval_end] Concise label
"""

    # Run LLaMA
    output = run_llama(prompt)
    if output:
        print(f"LLaMA raw output for grouping:\n{output}")
        grouped_chapters = parse_llama_output(output)
        if not grouped_chapters:
            print("No valid grouped chapters returned by LLaMA.")
            grouped_chapters = []
    else:
        print("No output from LLaMA for grouping.")
        grouped_chapters = []

    return grouped_chapters


def get_video_duration(video_path):
    command = ['ffprobe', '-v', 'error', '-show_entries',
               'format=duration', '-of',
               'default=noprint_wrappers=1:nokey=1', video_path]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return float(result.stdout.strip())


def process_video(video_path):
    print(f"Processing video: {video_path}")

    total_duration = get_video_duration(video_path)
    print(f"Total video duration: {total_duration} seconds")

    chunk_duration = 1200  # 20 minutes in seconds
    start_times = np.arange(0, total_duration, chunk_duration)

    all_chapters = []

    for i, start_time in enumerate(start_times):
        print(f"Processing chunk {i+1}/{len(start_times)} starting at {start_time} seconds")

        # Calculate the duration for this chunk
        duration = min(chunk_duration, total_duration - start_time)

        # Step 1: Extract audio for this chunk
        audio_data = extract_audio_ffmpeg(video_path, start_time, duration)

        if not audio_data:
            print(f"Failed to extract audio for chunk starting at {start_time}s. Skipping this chunk.")
            continue

        # Step 2: Transcribe the extracted audio
        segments = transcribe_audio(audio_data, start_time)

        if not segments:
            print(f"No transcription segments were generated for chunk starting at {start_time}s. Skipping this chunk.")
            continue

        # Step 3: Label segments
        labeled_intervals = label_intervals_with_ollama(segments, start_time, duration)

        # Step 4: Group labeled intervals into larger chapters
        grouped_chapters = group_labeled_intervals_with_llama(labeled_intervals)

        # Collect chapters
        all_chapters.extend(grouped_chapters)

    # After processing all chunks, add chapters to the video
    print("Generated grouped chapters:")
    for chapter in all_chapters:
        start_str = str(timedelta(seconds=int(chapter['start'])))
        end_str = str(timedelta(seconds=int(chapter['end'])))
        print(f"[{start_str}] [{end_str}] {chapter['label']}")

    add_chapters_to_video(video_path, all_chapters)


if __name__ == "__main__":
    video_file_path = sys.argv[1]
    print(f"Video file path received: {video_file_path}")
    process_video(video_file_path)
