#!/bin/bash
model="/llm/models/Llama-3.2-3B-Instruct-AWQ"
served_model_name="Llama-3.2-3B-Instruct-AWQ"

export CCL_WORKER_COUNT=2
export SYCL_CACHE_PERSISTENT=1
export FI_PROVIDER=shm
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=sockets
export CCL_ATL_SHM=1
 
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
export TORCH_LLM_ALLREDUCE=0

export CCL_SAME_STREAM=1
export CCL_BLOCKING_WAIT=0
 
source /opt/intel/1ccl-wks/setvars.sh

python -m ipex_llm.vllm.xpu.entrypoints.openai.api_server \
  --served-model-name $served_model_name \
  --port 8000 \
  --model $model \
  --trust-remote-code \
  --block-size 8 \
  --gpu-memory-utilization 0.95 \
  --device xpu \
  --dtype float16 \
  --enforce-eager \
  --quantization awq \
  --load-in-low-bit asym_int4 \
  --max-model-len 2000 \
  --max-num-batched-tokens 3000 \
  --max-num-seqs 256 \
  --tensor-parallel-size 2 \
  --disable-async-output-proc \
  --distributed-executor-backend ray
