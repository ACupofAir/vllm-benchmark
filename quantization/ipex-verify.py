from vllm import SamplingParams
from ipex_llm.vllm.xpu.engine import IPEXLLMClass as LLM

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
llm = LLM(model="/llm/models/Llama-2-7b-chat-hf/",
          device="xpu",
          dtype="float16",
          enforce_eager=True,
          load_in_low_bit="sym_int4",
          tensor_parallel_size=1,
          trust_remote_code=True)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
