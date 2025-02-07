from vllm import SamplingParams
from ipex_llm.vllm.xpu.engine import IPEXLLMClass as LLM
import sys

model_path = sys.argv[1]
tp_num = int(sys.argv[2])

# Sample prompts.
prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        ]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model=model_path,
          device="xpu",
          dtype="float16",
          enforce_eager=True,
          load_in_low_bit="fp6",
          tensor_parallel_size=tp_num,
          disable_async_output_proc=True,
          distributed_executor_backend="ray",
          max_model_len=2000,
          trust_remote_code=True,
          block_size=8,
          max_num_batched_tokens=2000)
print(llm)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
