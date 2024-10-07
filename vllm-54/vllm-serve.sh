#!/bin/bash
model="/llm/models/Qwen1.5-14B-Chat/"
served_model_name="Qwen1.5-14B-Chat"

#source /opt/intel/1ccl-wks/setvars.sh
export no_proxy=localhost,127.0.0.1
#export FI_PROVIDER=tcp
#export OMP_NUM_THREADS=48

#export SYCL_CACHE_PERSISTENT=1
#export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0
#export USE_XETLA=OFF

#export USE_XETLA=OFF
#export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0

#if [[ $KERNEL_VERSION != *"6.5"* ]]; then
#    export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
#fi
#export TORCH_LLM_ALLREDUCE=0
#export VLLM_USE_RAY_SPMD_WORKER=1
#export VLLM_USE_RAY_COMPILED_DAG=1
#export VLLM_USE_RAY_COMPILED_DAG_NCCL_CHANNEL=0

export LD_LIBRARY_PATH=/opt/intel/oneapi/tbb/2021.12/env/../lib/intel64/gcc4.8:/opt/intel/oneapi/mpi/2021.12/opt/mpi/libfabric/lib:/opt/intel/oneapi/mpi/2021.12/lib:/opt/intel/oneapi/mkl/2024.1/lib:/opt/intel/oneapi/ippcp/2021.11/lib/:/opt/intel/oneapi/ipp/2021.11/lib:/opt/intel/oneapi/dpl/2022.5/lib:/opt/intel/oneapi/dnnl/2024.1/lib:/opt/intel/oneapi/debugger/2024.1/opt/debugger/lib:/opt/intel/oneapi/dal/2024.2/lib:/opt/intel/oneapi/compiler/2024.1/opt/oclfpga/host/linux64/lib:/opt/intel/oneapi/compiler/2024.1/opt/compiler/lib:/opt/intel/oneapi/compiler/2024.1/lib:/opt/intel/oneapi/ccl/2021.12/lib/:/usr/local/lib/python3.11/dist-packages/torch/lib/:/usr/local/lib/python3.11/dist-packages/intel_extension_for_pytorch/lib/
source /opt/intel/oneapi/setvars.sh
source /opt/intel/1ccl-wks/setvars.sh

python -m vllm.entrypoints.openai.api_server \
  --served-model-name $served_model_name \
  --port 8000 \
  --model $model \
  --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --device xpu \
  --dtype float16 \
  --enforce-eager \
  --max-model-len 2048 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 256 \
  -pp 2
  #-tp 2 #--enable-prefix-caching --enable-chunked-prefill #--tokenizer-pool-size 8 --swap-space 8
