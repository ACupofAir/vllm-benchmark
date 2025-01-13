# TODO:
# * system_prompt
# * 

import time
import torch
from ipex_llm.transformers import AutoModelForCausalLM

system_prompt = "What can I do for you?"
model_path = ""
n_predict = 512

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_4bit=True,
    optimize_model=True,
    load_in_low_bit="fp8",
    trust_remote_code=True,
    use_cache=True,
)
model = model.half().to("xpu")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)




def run_batch_inference(data_batch):
    with torch.inference_mode():
        prompts = [f"User: {prompt}" for prompt in data_batch]
        full_prompts = [f"System: {system_prompt} \n{prompt}" for prompt in prompts]
        model_inputs = tokenizer(full_prompts, return_tensors="pt", padding=True).to(
            "xpu"
        )

        st = time.time()
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens = n_predict
        )
        torch.xpu.synchronize()
        
        generated_ids = generated_ids.cpu()
        generated_ids = [
            output_ids[len()]
        ]
