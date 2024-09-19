import os
import numpy as np
import gradio as gr

import torch
from ipex_llm import optimize_model
from ipex_llm.transformers import AutoModelForSpeechSeq2Seq, AutoModelForCausalLM
from transformers import WhisperProcessor, AutoTokenizer, AutoProcessor, BarkModel

WHISPER_MODEL_PATH = "/llm/models/whisper-tiny"
QWEN_MODEL_PATH = "/llm/models/Qwen1.5-14B-Chat"
BARK_MODEL_PATH = "/llm/models/bark-small"

# load model
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(WHISPER_MODEL_PATH,
                                                  load_in_4bit=True,
                                                  optimize_model=False,
                                                  use_cache=True)
whisper_model.to('xpu')
whisper_model.config.forced_decoder_ids = None

# load processor
whisper_processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_PATH)
forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(language="chinese", task="transcribe")


chat_history = []
qwen_model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_PATH,
                                                  load_in_4bit=True,
                                                  optimize_model=True,
                                                  trust_remote_code=True,
                                                  use_cache=True)
qwen_model = qwen_model.half().to('xpu')
qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_PATH,
                                               trust_remote_code=True)


bark_model = BarkModel.from_pretrained(BARK_MODEL_PATH)
bark_model = optimize_model(bark_model).to('xpu')
bark_processor = AutoProcessor.from_pretrained(BARK_MODEL_PATH)

def transcript(audio):
    sr, data = audio
    data = data.astype(np.float32)
    data /= np.max(np.abs(data))

    with torch.inference_mode():
        input_features = whisper_processor(data, sampling_rate=sr, return_tensors="pt").input_features.to('xpu')
        predicted_ids = whisper_model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
        transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription[0]

def chat(prompt):
    global chat_history
    print(chat_history)
    if chat_history == []:
        response, chat_history = qwen_model.chat(qwen_tokenizer, prompt, history=None)
    else:
        response, chat_history = qwen_model.chat(qwen_tokenizer, prompt, history=chat_history)

    return response

def tts(text):
    input = bark_processor(text).to('xpu')

    with torch.inference_mode():
        audio_array = bark_model.generate(**input)
        audio_array = audio_array.cpu().numpy().squeeze()

        sample_rate = bark_model.generation_config.sample_rate

    return (sample_rate, audio_array)

def work(audio):
    prompt = transcript(audio)
    print('======================DEBUG START: prompt======================')
    print(prompt)
    print('======================DEBUG  END : prompt======================')
    response = chat(prompt)
    answer = tts(response)

    return answer


input_audio = gr.Audio(
    sources=["microphone"],
    waveform_options=gr.WaveformOptions(
        waveform_color="#01C6FF",
        waveform_progress_color="#0066B4",
        skip_length=2,
        show_controls=False,
        sample_rate=16000
    ),
)
demo = gr.Interface(
    fn=work,
    inputs=input_audio,
    outputs="audio"
)

if __name__ == "__main__":
    demo.launch()