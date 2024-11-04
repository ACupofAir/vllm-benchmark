model="/llm/models/MiniCPM-V-2_6"
served_model_name="MiniCPM-V-2_6"
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
export TORCH_LLM_ALLREDUCE=0

# Tensor parallel related arguments:
export CCL_WORKER_COUNT=4
export SYCL_CACHE_PERSISTENT=1
# export CCL_WORKER_AFFINITY=12,13,14,15
# export CCL_WORKER_AFFINITY=""
export FI_PROVIDER=shm
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=sockets
export CCL_ATL_SHM=1

source /opt/intel/1ccl-wks/setvars.sh

# # Set CPU binding based on tensor_parallel_size
# if [ "4" -eq 1 ]; then
#   export CPU_BINDING=""
# elif [ "4" -eq 2 ]; then
#   export CPU_BINDING=""
# elif [ "4" -eq 4 ]; then
#   export CPU_BINDING=""
# else
#   export CPU_BINDING=""
# fi

# # Set NUMACTL_CMD based on whether CPU_BINDING is empty or not
# if [ -n "$CPU_BINDING" ]; then
#   export NUMACTL_CMD="numactl -C $CPU_BINDING"
# else
#   export NUMACTL_CMD=""
# fi

# Run the command, including numactl if set
python -m ipex_llm.vllm.xpu.entrypoints.openai.api_server \
	--port 8000 \
	--model $model \
	--served-model-name $served_model_name \
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
