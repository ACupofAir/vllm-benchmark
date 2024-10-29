import time
import torch
import sys

from ipex_llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

path = "/llm/models/gpt2-medium"
device = "xpu"
bsz = int(sys.argv[1])

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(path, load_in_low_bit="fp16", trust_remote_code=True)
model = model.half().eval()
print(model)

model = model.to(device)

prompt = "Once upon a time," * 204 + "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
input_ids = input_ids.to(device)

print(input_ids.shape)

with torch.inference_mode():
    for i in range(4):
        st = time.time()
        output_tokens = model.generate(input_ids, do_sample=False, max_new_tokens=1)
        torch.xpu.synchronize()
        et = time.time()
        print(et - st)

        # output_str = tokenizer.decode(output_tokens[0])
        # print(output_str)

prompts = ["Once upon a time," * 204 + "Once upon a time"] * bsz
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
