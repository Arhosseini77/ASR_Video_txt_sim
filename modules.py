import Levenshtein
import torch
from moviepy.editor import VideoFileClip
from transformers import pipeline


def calculate_similarity_score(video_path, text):
    """
    Calculate the similarity score between a given text and the transcribed speech from a video.

    Parameters:
    - video_path (str): The path to the video file for which you want to transcribe the speech.
    - text (str): The reference text that you want to compare the transcribed speech against.

    Returns:
    - similarity (float): A similarity score between 0.0 and 1.0, indicating how similar the transcribed speech
      is to the reference text. A higher score indicates greater similarity.

    This function performs the following steps:
    1. Extracts audio from the video file.
    2. Utilizes an Automatic Speech Recognition (ASR) model to transcribe the audio into text.
    3. Calculates the Levenshtein distance between the transcribed text and the reference text.
    4. Computes the similarity score

    Note:
    - The ASR model used in this function is 'openai/whisper-large-v3'.
    """
    # Extract Audio from Video
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile("temp_audio.wav")

    # Initialize the pipeline for ASR
    pipe = pipeline("automatic-speech-recognition",
                    "openai/whisper-large-v3",
                    torch_dtype=torch.float16,
                    device="cuda")

    # Perform ASR on the extracted audio
    asr_output = pipe("temp_audio.wav")["text"]

    # Calculate Levenshtein distance and similarity
    distance = Levenshtein.distance(text, asr_output)
    max_distance = max(len(text), len(asr_output))
    similarity = 1 - (distance / max_distance) if max_distance != 0 else 1.0

    return similarity
