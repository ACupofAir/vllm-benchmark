#!/bin/bash
model="/llm/models/Meta-Llama-3.1-70B-Instruct"
served_model_name="Meta-Llama-3.1-70B-Instruct"

# export SYCL_CACHE_PERSISTENT=1
export CCL_WORKER_COUNT=4
export FI_PROVIDER=shm
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=sockets
export CCL_ATL_SHM=1

export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
export TORCH_LLM_ALLREDUCE=0

source /opt/intel/1ccl-wks/setvars.sh

python -m ipex_llm.vllm.xpu.entrypoints.openai.api_server \
  --served-model-name $served_model_name \
  --port 8001 \
  --model $model \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --device xpu \
  --dtype float16 \
  --enforce-eager \
  --load-in-low-bit fp8 \
  --max-model-len 4000 \
  --max-num-batched-tokens 4000 \
  --max-num-seqs 12 \
  --api-key intel123 \
  --tensor-parallel-size 8 \
  --distributed-executor-backend ray
