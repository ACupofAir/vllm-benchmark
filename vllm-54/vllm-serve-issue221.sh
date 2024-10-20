#!/bin/bash
model="/llm/models/Qwen1.5-14B-Chat/"
served_model_name="Qwen1.5-14B-Chat"

export no_proxy=localhost,127.0.0.1

source /opt/intel/oneapi/setvars.sh
source /opt/intel/1ccl-wks/setvars.sh

python -m ipex_llm.vllm.xpu.entrypoints.openai.api_server \
  --served-model-name $served_model_name \
  --port 8001 \
  --model $model \
  --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --device xpu \
  --dtype float16 \
  --enforce-eager \
  --load-in-low-bit sym_int4 \
  --max-model-len 2048 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 256 \
  -tp 1 \
  --max-num-seqs 64
  #-tp 2 #--enable-prefix-caching --enable-chunked-prefill #--tokenizer-pool-size 8 --swap-space 8
