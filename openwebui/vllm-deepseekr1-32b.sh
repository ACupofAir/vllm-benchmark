#!/bin/bash
model="/llm/models/DeepSeek-R1-Distill-Qwen-32B"
served_model_name="DeepSeek-R1-Distill-Qwen-32B"

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
  --block-size 8 \
  --gpu-memory-utilization 0.95 \
  --device xpu \
  --dtype float16 \
  --enforce-eager \
  --load-in-low-bit sym_int4 \
  --max-model-len 4000 \
  --max-num-batched-tokens 4000 \
  --max-num-seqs 256 \
  --api-key intel123 \
  --tensor-parallel-size 2 \
  --disable-async-output-proc \
  --distributed-executor-backend ray
