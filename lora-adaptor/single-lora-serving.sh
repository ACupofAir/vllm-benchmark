export SYCL_CACHE_PERSISTENT=1
export SQL_LOARA=/llm/models/llama-2-7b-sql-lora-test
python -m ipex_llm.vllm.xpu.entrypoints.openai.api_server \
  --model /llm/models/Llama-2-7b-chat-hf \
  --served-model-name Llama-2-7b-chat-hf \
  --port 8001 \
  --trust-remote-code \
  --gpu-memory-utilization 0.75 \
  --device xpu \
  --dtype float16 \
  --enforce-eager \
  --load-in-low-bit fp8 \
  --max-model-len 4096 \
  --max-num-batched-tokens 10240 \
  --max-num-seqs 12 \
  --tensor-parallel-size 1 \
  --enable-lora \
  --distributed-executor-backend ray \
  --lora-modules sql-lora=$SQL_LOARA
