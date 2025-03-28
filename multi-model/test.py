import ipex_llm
import os
import time
import torch
import requests
import librosa
from diffusers import DiffusionPipeline
from PIL import Image
from ipex_llm.transformers import AutoModel
from ipex_llm.transformers import AutoModelForCausalLM
from ipex_llm.optimize import low_memory_init, load_low_bit
from ipex_llm import optimize_model
from ipex_llm.transformers import AutoModelForSpeechSeq2Seq
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    WhisperProcessor,
    AutoModelForSeq2SeqLM,
)

WHISPER_SAMPLING_RATE = 16000


def test_chatglm(llm_model, llm_tokenizer, report, is_report):
    with torch.inference_mode():
        # warm up=======================
        prompt = "问：<s>[INST]请简单地回答。 <<SYS>>\n\n\n<</SYS>>\n\n请介绍一下上海 [/INST]\n\n答："
        # tokenize the input prompt from string to token ids;
        # with .to('xpu') specifically for inference on Intel GPUs
        input_ids = llm_tokenizer.encode(prompt, return_tensors="pt").to("xpu")
        st = time.time()
        # predict the next tokens (maximum 32) based on the input token ids
        output = llm_model.generate(input_ids, max_new_tokens=128)
        if is_report:
            report.append("llm infer {:.2f} s".format(time.time() - st))
        print(f"==============================llm generate {time.time()-st}")
        # decode the predicted token ids to output string

        st = time.time()
        output = output.cpu()
        output_str = llm_tokenizer.decode(output[0], skip_special_tokens=True)
        # output = output.cpu()
        print(f"==============================llm decode {time.time()-st}")
        print("-" * 20, "Output", "-" * 20)
        print(output_str)


def test_sd(sd_model, report, is_report):
    imgidx = 1
    save_path = "/home/arda/junwang/imgs" + str(imgidx) + ".png"
    prompt_en = "a pig flying in the sky"
    prompt_en2 = "a horse running in the city"

    st = time.time()
    output_image = sd_model(
        prompt=prompt_en2,
        num_inference_steps=4,
        guidance_scale=1,
        num_images_per_prompt=1,
        output_type="pil",
        width=512,
        height=512,
        lcm_origin_steps=50,
    ).images[0]

    if is_report:
        report.append("sd  infer {:.2f} s".format(time.time() - st))
    print(
        "========== sd generate {:.2f} s".format(time.time() - st)
    )  # 0.51s, Arc770, 16GB
    output_image.save(save_path)
    print("========== sd save {:.2f} s".format(time.time() - st))  # 0.64s, Arc770, 16GB


def test_minicpm(model, tokenizer, report, is_report):
    query = "图片里有什么?他们在干什么？"
    stream = False
    image_path = "/home/arda/junwang/imgs/basketball.jpg"
    if os.path.exists(image_path):
        image = Image.open(image_path).convert("RGB")
    else:
        image = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
    msgs = [{"role": "user", "content": [image, query]}]
    model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer,
    )

    if stream:
        res = model.chat(image=None, msgs=msgs, tokenizer=tokenizer, stream=True)

        print("-" * 20, "Input Image", "-" * 20)
        print(image_path)
        print("-" * 20, "Input Prompt", "-" * 20)
        print(query)
        print("-" * 20, "Stream Chat Output", "-" * 20)
        for new_text in res:
            print(new_text, flush=True, end="")
    else:
        st = time.time()
        res = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
        )
        torch.xpu.synchronize()
        end = time.time()
        if is_report:
            report.append("cpm infer {:.2f} s".format(end - st))
        print(f"=========minicpm Inference time: {end-st} s")
        print("-" * 20, "Input Image", "-" * 20)
        print(image_path)
        print("-" * 20, "Input Prompt", "-" * 20)
        print(query)
        print("-" * 20, "Chat Output", "-" * 20)
        print(res)


def test_whisper(whisper_processor, whisper_model, report, is_report):
    init_wav_file = "/home/arda/junwang/audios/Recording.wav"
    audio, __ = librosa.load(path=init_wav_file, sr=WHISPER_SAMPLING_RATE)
    simple_zh = "以下是普通话的句子[\INST]"  # 简体字，否则会出现繁体字
    prompt_ids = whisper_processor.get_prompt_ids(simple_zh)
    forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(
        language="chinese", task="transcribe"
    )
    output_str = None
    with torch.inference_mode():
        if device == "GPU":
            input_features = (
                whisper_processor(
                    audio, sampling_rate=WHISPER_SAMPLING_RATE, return_tensors="pt"
                )
                .input_features.half()
                .to("xpu")
            )
        else:
            input_features = whisper_processor(
                audio, sampling_rate=WHISPER_SAMPLING_RATE, return_tensors="pt"
            ).input_features
        st = time.time()
        predicted_ids = whisper_model.generate(
            input_features, forced_decoder_ids=forced_decoder_ids, prompt_ids=prompt_ids
        )
        output_str = whisper_processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        output_str = output_str[0].split("[\INST]")[-1]
    inf_time = time.time() - st
    if is_report:
        report.append("wsp infer {:.2f} s".format(inf_time))
    print("=============== whisper latency {:.2f} s".format(inf_time))
    print("-" * 20, "Recognized text", "-" * 20)
    print(f"{output_str}")
    print("-" * 40)
    return output_str


if __name__ == "__main__":
    # TODO:[config]
    device = "GPU"
    llm_test = True
    sd_test = True
    minicpm_test = True
    wp_test = True
    metric_report = []
    test_loop = 10

    #############
    # load whisper
    #############
    if wp_test:
        st = time.time()
        whisper_model_path = "/home/arda/LLM/whisper-small"

        # Load model in 4 bit,
        # which convert the relevant layers in the model into INT4 format
        if device == "GPU":
            whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                whisper_model_path,
                load_in_4bit=True,
                optimize_model=False,
                use_cache=True,
            )
            whisper_model.half().to("xpu")
        else:
            whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                whisper_model_path, load_in_4bit=True
            )
        whisper_model.config.forced_decoder_ids = None

        # Load processor
        whisper_processor = WhisperProcessor.from_pretrained(whisper_model_path)
        print("============ whisper load to gpu")
        # test_whisper(whisper_processor, whisper_model, metric_report, False)
        print("============ whisper warm up done!")

    #############
    # load chatglm
    #############
    if llm_test:
        # dummy_elements_num = 490 * 1024 * 1024 // 4
        # dummy_tensor = torch.zeros(dummy_elements_num, dtype=torch.float32, device="xpu")
        llm_model_path = "/home/arda/LLM/chatglm3-6b"
        save_directory = "/home/arda/junwang/model_low_bit/chatglm3-6b"
        llm_tokenizer = AutoTokenizer.from_pretrained(
            llm_model_path, trust_remote_code=True
        )
        if os.path.exists(save_directory):
            llm_model = AutoModel.load_low_bit(save_directory, trust_remote_code=True)
        else:
            llm_model = AutoModel.from_pretrained(
                llm_model_path,
                load_in_4bit=True,
                optimize_model=True,
                trust_remote_code=True,
                use_cache=True,
            )
            llm_model.save_low_bit(save_directory)
            del llm_model
            llm_model = AutoModel.load_low_bit(save_directory, trust_remote_code=True)

        if device == "GPU":
            llm_model.to("xpu")
        print("============ llm load to gpu")
        # test_chatglm(llm_model, llm_tokenizer, metric_report, False)
        print("============ llm warm done!")

    ##########################
    # load lcm stable diffusion
    ##########################
    if sd_test:
        sd_model_path = "/home/arda/LLM/LCM_Dreamshaper_v7"
        sd_model = DiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=sd_model_path, torch_dtype=torch.float16
        )
        sd_model.to("xpu")
        print("============ sd load to gpu")
        # test_sd(sd_model, metric_report, False)
        print("============ sd warm done!")

    #############
    # load minicpm
    #############
    if minicpm_test:
        minicpm_model_path = "/home/arda/LLM/MiniCPM-V-2_6"
        minicpm_lowbit_path = "/home/arda/junwang/model_low_bit/MiniCPM-V-2_6"

        if not minicpm_lowbit_path or not os.path.exists(minicpm_lowbit_path):
            minicpm_model = AutoModel.from_pretrained(
                minicpm_model_path,
                load_in_low_bit="sym_int4",
                optimize_model=True,
                trust_remote_code=True,
                use_cache=True,
                modules_to_not_convert=["vpm", "resampler"],
            )
            minicpm_tokenizer = AutoTokenizer.from_pretrained(
                minicpm_model_path, trust_remote_code=True
            )
        else:
            minicpm_model = AutoModel.load_low_bit(
                minicpm_lowbit_path,
                optimize_model=True,
                trust_remote_code=True,
                use_cache=True,
                modules_to_not_convert=["vpm", "resampler"],
            )
            minicpm_tokenizer = AutoTokenizer.from_pretrained(
                minicpm_lowbit_path, trust_remote_code=True
            )

        minicpm_model.eval()

        if minicpm_model_path and not os.path.exists(minicpm_model_path):
            minicpm_processor = AutoProcessor.from_pretrained(
                minicpm_model_path, trust_remote_code=True
            )
            minicpm_model.save_low_bit(minicpm_lowbit_path)
            minicpm_tokenizer.save_pretrained(minicpm_lowbit_path)
            minicpm_processor.save_pretrained(minicpm_lowbit_path)

        minicpm_model = minicpm_model.half().to("xpu")
        print("============ cpm load to gpu")
        # test_minicpm(minicpm_model, minicpm_tokenizer, metric_report, False)
        print("============ cpm warm up done!")

    #############
    # test
    #############
    i = 0
    while i < test_loop:
        i += 1
        print("\n", "==" * 20, " TEST STARTING ", "==" * 20)
        metric_report.append("\n-----------------loop {}".format(i))
        if llm_test:
            print("\n=================test_chatglm")
            test_chatglm(llm_model, llm_tokenizer, metric_report, True)
        if minicpm_test:
            print("\n=================test_minicpm")
            test_minicpm(minicpm_model, minicpm_tokenizer, metric_report, True)
        if sd_test:
            print("\n=================test_sd")
            test_sd(sd_model, metric_report, True)
        if wp_test:
            print("\n=================test_whisper")
            test_whisper(whisper_processor, whisper_model, metric_report, True)

    for rp in metric_report:
        print(rp)
