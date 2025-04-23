set -x
LLAMA_SERVER=/home/intel/junwang/llama-cpp-ipex-llm-2.2.0-ubuntu-xeon/llama-server
MODEL=/home/intel/LLM/DeepSeek-R1-GGUF/DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf
NAME=DeepSeek-R1-Q4_K_M

export SYCL_CACHE_PERSISTENT=1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export IPEX_LLM_SCHED_MAX_COPIES=1
export IPEX_LLM_QUANTIZE_KV_CACHE=1 # to enable fp8 kv cache
export ONEAPI_DEVICE_SELECTOR=level_zero:0,1
CORES=$(lscpu | grep "Core(s) per socket:" | awk '{print $4}')
$LLAMA_SERVER -m $MODEL -t $CORES -e -ngl 999 --no-context-shift -ot exps=CPU --host 0.0.0.0 --port 8001 --alias $NAME
