set -x
source /opt/intel/oneapi/setvars.sh --force

export SYCL_CACHE_PERSISTENT=1
export ONEAPI_DEVICE_SELECTOR=level_zero:0,1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export IPEX_LLM_SCHED_MAX_COPIES=1
export IPEX_LLM_QUANTIZE_KV_CACHE=1
export KMP_AFFINITY="granularity=fine,verbose,compact,1,0"

llama_cli_path=/home/llm/junwang/llama-cpp-ipex-llm-2.3.0b20250430-ubuntu-xeon/llama-server
#llama_cli_path=/home/llm/junwang/llama-cpp-ipex-llm-2.3.0b20250430-ubuntu-xeon/llama-cli
model_path=/home/llm/shane/DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf

ngl=99
thread=35 # change to two socket threads
ot="exps=CPU"

#numa="numactl --interleave=0-1"
numa="numactl -N 0 -m 0"
#$numa $llama_cli_path -m $model_path --no-context-shift -no-cnv -n 300 --prompt "Whats AI?" -t $thread -e -ngl $ngl --color -c 2048 --temp 0 -ot $ot
$numa $llama_cli_path -m $model_path --no-context-shift -t $thread -e -ngl $ngl -c 2048 --temp 0 -ot $ot --host 0.0.0.0 --port 8001 --jinja --chat-template-file template.jinja --verbose
