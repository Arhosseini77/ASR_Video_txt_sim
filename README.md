## Intro
This module check Similarity Between input Text vs Audio of input Video 
using Whisper ASR and Levenshtein for checking similarity

## Install 

* Python: 3.10 , Torch :  2.1.1 , Cuda : 11.8
 
```python
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r req.txt
```

## Example

Modify main.py 
```python
# Define the text and video path
text = "Your text here"
video_path = "path/to/your/video.mp4"
```
then run main.py
