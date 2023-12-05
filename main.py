from modules import calculate_similarity_score


def main():
    """
    Calculate the similarity score between a reference text and the ASR speech from a video.
    """
    # Define the text and video path
    text = "Your reference text here"
    video_path = "path/to/your/video.mp4"

    # Calculate the similarity score
    similarity_score = calculate_similarity_score(video_path, text)
    print(f"Similarity probability: {similarity_score:.2f}")


if __name__ == "__main__":
    main()
