set -x
source /opt/intel/oneapi/setvars.sh

CORES=43
LLAMA_CLI=/home/intel/junwang/llama-cpp-bigdl/build_4_27/bin/llama-cli
MODEL=/home/intel/models/DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf
PROMPT=/home/intel/junwang/input.txt

export SYCL_CACHE_PERSISTENT=1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export IPEX_LLM_SCHED_MAX_COPIES=1
export IPEX_LLM_QUANTIZE_KV_CACHE=1 # to enable fp8 kv cache
export FLASH_MOE_EP=1
export KMP_AFFINITY="granularity=fine,proclist=[0,2-$CORES],explicit"
export KMP_BLOCKTIME=200
export ONEAPI_DEVICE_SELECTOR=level_zero:0

$LLAMA_CLI -m $MODEL -f $PROMPT -n 128 -t $CORES -e -ngl 999 --color -c 2500 --no-context-shift -ot exps=CPU
