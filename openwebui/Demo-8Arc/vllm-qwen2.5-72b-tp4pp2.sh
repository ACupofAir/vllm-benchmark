#!/bin/bash
model="/llm/models/LLM/Qwen2.5-72B-Instruct"
served_model_name="Qwen2.5-72B-8xARC770-IPEX-LLM-vLLM"

export VLLM_ENGINE_ITERATION_TIMEOUT_S=600
# export SYCL_CACHE_PERSISTENT=1
export CCL_WORKER_COUNT=8
#export FI_PROVIDER=shm
#export CCL_ATL_TRANSPORT=ofi
#export CCL_ZE_IPC_EXCHANGE=sockets
#export CCL_ATL_SHM=1

export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
export TORCH_LLM_ALLREDUCE=0

source /opt/intel/1ccl-wks/setvars.sh

python -m ipex_llm.vllm.xpu.entrypoints.openai.api_server \
    --served-model-name $served_model_name \
    --port 8001 \
    --model $model \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --device xpu \
    --dtype float16 \
    --enforce-eager \
    --load-in-low-bit fp8 \
    --max-model-len 3000 \
    --max-num-batched-tokens 3000 \
    --max-num-seqs 256 \
    --block-size 8 \
    --distributed-executor-backend ray \
    --disable-async-output-proc \
    --api-key intel123 \
    -pp 2 \
    --tensor-parallel-size 4
