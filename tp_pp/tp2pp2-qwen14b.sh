export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
export TORCH_LLM_ALLREDUCE=0

# Tensor parallel related arguments:
export CCL_WORKER_COUNT=4
export SYCL_CACHE_PERSISTENT=1
# export CCL_WORKER_AFFINITY=12,13,14,15
# export CCL_WORKER_AFFINITY=""
#export FI_PROVIDER=shm
#export CCL_ATL_TRANSPORT=ofi
#export CCL_ZE_IPC_EXCHANGE=sockets
#export CCL_ATL_SHM=1

source /opt/intel/1ccl-wks/setvars.sh

# Run the command, including numactl if set
python -m ipex_llm.vllm.xpu.entrypoints.openai.api_server \
	--port 8000 \
	--model "/llm/models/Qwen1.5-14B-Chat" \
	--served-model-name "Qwen1.5-14B-Chat" \
	--trust-remote-code \
	--gpu-memory-utilization "0.9" \
	--device xpu \
	--dtype float16 \
	--enforce-eager \
	--load-in-low-bit "fp8" \
	--max-model-len "2048" \
	--max-num-batched-tokens "2048" \
	--max-num-seqs "256" \
	--tensor-parallel-size "2" \
	--pipeline-parallel-size "1" \
	--block-size 8 \
	--distributed-executor-backend ray
