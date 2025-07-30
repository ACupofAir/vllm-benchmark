ONEAPI=2025.0
LLM_MODELS=/llm/models
source /opt/intel/oneapi/$ONEAPI/oneapi-vars.sh --force

export ONEAPI_DEVICE_SELECTOR=level_zero:0,1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1

# [RELEASE VERSION]
llama_cli=/llm/workspace/llama-cpp-bigdl/build-$ONEAPI/bin/llama-cli

# [DEBUG   VERSION]
#cd /llm/workspace/llama-cpp-bigdl/build/bin/
#llama_cli=/llm/workspace/llama-cpp-bigdl/build/bin/llama-cli

gguf_path=$LLM_MODELS/DeepSeek-R1-UD-IQ1_S/DeepSeek-R1-UD-IQ1_S-00001-of-00003.gguf
#gguf_path=$LLM_MODELS/DeepSeek-R1-Distill-Qwen-32B-Q4_0.gguf
prompt_file=/llm/workspace/vllm-benchmark/prompts/1024.txt

ot="exps=CPU"
ngl=999
thread=23
ctx=1300

$llama_cli \
    -m $gguf_path \
    -f $prompt_file \
    -n 100 \
    -c $ctx \
    -t $thread \
    -e \
    -ngl $ngl \
    --color \
    --temp 0 \
    -no-cnv \
    --no-warmup \
    -sm hybrid \
    -ot $ot
