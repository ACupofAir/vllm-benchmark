#  TODO:
# * system_prompt
# * model
# * tokenizer

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ipex_llm import optimize_model

system_prompt = "What can I do for you?"
model_path = "/home/arda/LLM/chatglm3-6b"
n_predict = 32

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype="auto",
    low_cpu_mem_usage=True,
    use_cache=True,
)
model = optimize_model(model)
model = model.to("xpu")

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
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=n_predict)
        torch.xpu.synchronize()

        generated_ids = generated_ids.cpu()
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        end = time.time()

        return responses, end - st


if __name__ == "__main__":
    data_batch_1 = ["What is the weather today?"]
    data_batch_2 = [
        "What is the weather today?",
        "Tell me a joke.",
    ]
    data_batch_4 = [
        "What is the weather today?",
        "Tell me a joke.",
        "What is the capital of France?",
        "How do I make pancakes?",
    ]

    batch_sizes = [1, 2, 4]
    data_batches = [data_batch_1, data_batch_2, data_batch_4]

for batch_size, data_batch in zip(batch_sizes, data_batches):
    responses, inference_time = run_batch_inference(data_batch)
    print(f"Batch size: {batch_size}")
    print(f"Responses: {responses}")
    print(f"Inference time: {inference_time:.2f} seconds\n")
