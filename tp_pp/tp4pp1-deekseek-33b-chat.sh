#!/bin/bash
model="/llm/models/DeepSeek-V2-Chat"
served_model_name="DeepSeek-v2-Chat"

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
	--gpu-memory-utilization 0.9 \
	--device xpu \
	--dtype float16 \
	--enforce-eager \
	--load-in-low-bit fp8 \
	--max-model-len 4000 \
	--max-num-batched-tokens 4000 \
	--max-num-seqs 12 \
	--tensor-parallel-size 4 \
	--disable-async-output-proc \
	--distributed-executor-backend ray

