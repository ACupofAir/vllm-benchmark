import time
import torch

from ipex_llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

path = "/llm/models/Qwen2.5-0.5B-Instruct"
device = "xpu"

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, load_in_low_bit="fp16", trust_remote_code=True)
model = model.half().eval()
print(model)

model = model.to(device)

prompt = "Once upon a time," * 100
input_ids = tokenizer.encode(prompt, return_tensors="pt")
print(input_ids.shape)

bsz = 8
prompts = ["Once upon a time," * 100] * bsz
inputs = tokenizer(prompts, return_tensors="pt")
inputs = inputs.to(device)

with torch.inference_mode():
    for i in range(4):
        st = time.time()
        output_tokens = model.generate(**inputs, do_sample=False, max_new_tokens=1)
        torch.xpu.synchronize()
        et = time.time()
        print(et - st)

        # output_str = tokenizer.decode(output_tokens[0])
        # print(output_str)
