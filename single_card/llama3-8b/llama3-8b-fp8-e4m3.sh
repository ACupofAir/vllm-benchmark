#!/bin/bash

# Chatglm3 benchmark
bash /llm/enable_sdpa.sh
export SYCL_CACHE_PERSISTENT=1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0
export USE_XETLA=OFF
export MODEL="/llm/models/Meta-Llama-3-8B-Instruct/"

# Benchmark config for prompts
export NUM_PROMPTS=145
export IN_LEN=1024
export OUT_LEN=512

# vLLM config
export LOW_BIT="fp8_e4m3"
export MAX_NUM_BATHCED_TOKENS=2000
export MAX_MODEL_LEN=2000
export MAX_NUM_SEQS=29
export TENSOR_PARALLEL_SIZE=1
export GPU_UTILIZATION_RATE=0.96

python /llm/benchmark_vllm_throughput.py --backend vllm --model $MODEL --num-prompts $NUM_PROMPTS --input-len $IN_LEN --output-len $OUT_LEN --trust-remote-code --enforce-eager --dtype float16 --device xpu --load-in-low-bit $LOW_BIT --gpu-memory-utilization $GPU_UTILIZATION_RATE --max-model-len $MAX_MODEL_LEN --max-num-batched-tokens $MAX_NUM_BATHCED_TOKENS --max-num-seqs $MAX_NUM_SEQS --tensor-parallel-size 1
