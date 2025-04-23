source /home/intel/oneapi/setvars.sh --force # change to related oneapi path

export SYCL_CACHE_PERSISTENT=1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export IPEX_LLM_SCHED_MAX_COPIES=1
export IPEX_LLM_QUANTIZE_KV_CACHE=1
export ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3
export KMP_AFFINITY="granularity=fine,verbose,compact,1,0"

llama_cli_path=/home/intel/junwang/llama-cpp-ipex-llm-2.2.0-ubuntu-xeon/llama-cli
model_path=/home/intel/LLM/DeepSeek-R1-GGUF/DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf
prompt_file=/home/intel/junwang/vllm-benchmark/prompts/1024.txt
log_file=/home/intel/junwang/vllm-benchmark/vtune/logs
ngl=99
thread=60
log_file="${log_file}/ds_${ngl}_${thread}.log"

touch $log_file
echo log_file: $log_file

ot="exps=CPU"
collect="hotspots"

vtune -collect $collect -- $llama_cli_path -m $model_path -no-cnv --no-context-shift -n 128 -f $prompt_file -t $thread -e -ngl $ngl --color -c 2048 --temp 0 -ot $ot
# >> $log_file 2>&1
