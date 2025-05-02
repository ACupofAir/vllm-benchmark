set -x
source /opt/intel/oneapi/setvars.sh

CORES=43
LLAMA_CLI=/home/intel/junwang/llama-cpp-bigdl/build_4_27/bin/llama-server
MODEL=/home/intel/models/DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf
PROMPT=/home/intel/junwang/input.txt
HOST=0.0.0.0
PORT=8002

export no_proxy=127.0.0.1,localhost
export SYCL_CACHE_PERSISTENT=1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export IPEX_LLM_SCHED_MAX_COPIES=1
export IPEX_LLM_QUANTIZE_KV_CACHE=1 # to enable fp8 kv cache
export FLASH_MOE_EP=1
export KMP_AFFINITY="granularity=fine,proclist=[0,2-$CORES],explicit"
export KMP_BLOCKTIME=200
export ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3

$LLAMA_CLI -m $MODEL  -t $CORES -e -ngl 999 -c 29000 --no-context-shift -ot exps=CPU  --host $HOST --port $PORT --alias DeepSeek-R1-Q4_K_M
