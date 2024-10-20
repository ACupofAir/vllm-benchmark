#!/bin/bash
model="/llm/models/Qwen1.5-14B-Chat"
served_model_name="Qwen1.5-14B-Chat"

export CCL_WORKER_COUNT=4
# export FI_PROVIDER=shm
# export CCL_ATL_TRANSPORT=ofi
# export CCL_ZE_IPC_EXCHANGE=sockets
# export CCL_ATL_SHM=1
 
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
export TORCH_LLM_ALLREDUCE=0
 
source /opt/intel/1ccl-wks/setvars.sh

python -m ipex_llm.vllm.xpu.entrypoints.openai.api_server \
  --served-model-name $served_model_name \
  --port 8000 \
  --model $model \
  --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --device xpu \
  --dtype float16 \
  --enforce-eager \
  --load-in-low-bit fp8 \
  --max-model-len 3000 \
  --max-num-batched-tokens 3000 \
  --max-num-seqs 12 \
  -tp 2 \
  -pp 2 
