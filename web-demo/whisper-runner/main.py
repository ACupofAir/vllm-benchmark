import gradio as gr
from transformers import pipeline
import numpy as np

transcriber = pipeline("automatic-speech-recognition", model="/llm/models/whisper-tiny/")

def transcribe(audio):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    return transcriber({"sampling_rate": sr, "raw": y})["text"]  

demo = gr.Interface(
    transcribe,
    gr.Audio(sources=["microphone"]),
    "text",
)

demo.launch()