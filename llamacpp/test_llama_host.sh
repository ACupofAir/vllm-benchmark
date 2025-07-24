source /opt/intel/oneapi/setvars.sh --force
ONEAPI=2025.1
LLM_MODELS=/home/arda/llm-models
#source ~/intel/oneapi/$ONEAPI/oneapi-vars.sh --force

export ONEAPI_DEVICE_SELECTOR=level_zero:0,1
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1

cd /home/arda/junwang/llama-cpp-bigdl/build-$ONEAPI/bin
llama_cli=/home/arda/junwang/llama-cpp-bigdl/build-$ONEAPI/bin/llama-cli

#gguf_path=$LLM_MODELS/qwen2-7b-instruct-q4_0.gguf
#gguf_path=$LLM_MODELS/DeepSeek-R1-Distill-Qwen-32B-Q4_0.gguf
gguf_path=$LLM_MODELS/DeepSeek-R1-UD-IQ1_S/DeepSeek-R1-UD-IQ1_S-00001-of-00003.gguf
prompt_file=/home/arda/junwang/vllm-benchmark/prompts/1024.txt

ot="exps=CPU"
ngl=999
thread=23
ctx=1300

$llama_cli \
    -m $gguf_path \
    -p "Once upon a time, there existed a little girl who" \
    -n 10 \
    -c $ctx \
    -t $thread \
    -e \
    -ngl $ngl \
    --color \
    --temp 0 \
    --no-warmup \
    -no-cnv \
    -sm hybrid \
    -ot $ot

#    -f $prompt_file \
