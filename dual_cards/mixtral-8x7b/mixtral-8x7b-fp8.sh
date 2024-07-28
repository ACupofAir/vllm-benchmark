#!/bin/bash

# Chatglm3 benchmark
export SYCL_CACHE_PERSISTENT=1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0
export USE_XETLA=OFF
export MODEL="/llm/models/Mixtral-8x7B-Instruct-v0.1"

# Benchmark config for prompts
export NUM_PROMPTS=2
export IN_LEN=1024
export OUT_LEN=512

# CCL related configs
export TORCH_LLM_ALLREDUCE=0
export CCL_DG2_ALLREDUCE=1

# Tensor parallel related arguments:
export CCL_WORKER_COUNT=4
export FI_PROVIDER=shm
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=sockets
export CCL_ATL_SHM=1

source /opt/intel/1ccl-wks/setvars.sh

# vLLM config
export LOW_BIT="sym_int4"
export MAX_NUM_BATHCED_TOKENS=2000
export MAX_MODEL_LEN=2000
export MAX_NUM_SEQS=1
export TENSOR_PARALLEL_SIZE=4
export GPU_UTILIZATION_RATE=0.6

python /llm/benchmark_vllm_throughput.py --backend vllm --model $MODEL --num-prompts $NUM_PROMPTS --input-len $IN_LEN --output-len $OUT_LEN --trust-remote-code --enforce-eager --dtype float16 --device xpu --load-in-low-bit $LOW_BIT --gpu-memory-utilization $GPU_UTILIZATION_RATE --max-model-len $MAX_MODEL_LEN --max-num-batched-tokens $MAX_NUM_BATHCED_TOKENS --max-num-seqs $MAX_NUM_SEQS --tensor-parallel-size $TENSOR_PARALLEL_SIZE
