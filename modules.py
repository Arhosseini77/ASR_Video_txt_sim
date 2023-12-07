import torch
from moviepy.editor import VideoFileClip
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity_score(video_path, text):
    """
    Calculate the similarity score between a given text and the transcribed speech from a video.

    This function performs the following steps:
    1. Extracts audio from the video file.
    2. Utilizes an Automatic Speech Recognition (ASR) model to transcribe the audio into text.
    3. Creates bigram (2-gram) representations of both the transcribed text and the reference text.
    4. Calculates the cosine similarity between these bigram representations.

    Parameters:
    - video_path (str): The path to the video file for which you want to transcribe the speech.
    - text (str): The reference text that you want to compare the transcribed speech against.

    Returns:
    - similarity (float): A similarity score between 0.0 and 1.0, indicating how similar the transcribed speech
      is to the reference text. A higher score indicates greater similarity.

    The use of bigram representations allows this function to be sensitive to the order of words,
    providing a more nuanced comparison than simple word or character-level analyses.

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

    # Create n-gram representation
    vectorizer = CountVectorizer(ngram_range=(2, 2))
    ngram_matrix = vectorizer.fit_transform([text, asr_output])

    # Calculate cosine similarity
    similarity_score = cosine_similarity(ngram_matrix[0:1], ngram_matrix[1:2])[0][0]

    return similarity_score
