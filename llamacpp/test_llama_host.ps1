$env:SYCL_CACHE_PERSISTENT=1
$env:SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
$env:ENABLE_PEER_ALLREDUCE=0
$env:ONEAPI_DEVICE_SELECTOR="level_zero:0,1"

$llama_dir="C:\Users\ADC\workspace\llama-cpp-bigdl"
$prompt="C:\Users\ADC\workspace\vllm-benchmark\prompts\1024.txt"
$llama_cli="$llama_dir\build\bin\llama-cli.exe"

& $llama_cli -m "C:\Users\ADC\workspace\models\DeepSeek-R1-Distill-Qwen-32B-Q4_0.gguf" `
    -f $prompt `
    -n 128 `
    -t 16 `
    -e `
    -ngl 99 `
    --color `
    --ctx-size 1200 `
    --temp 0  `
    -no-cnv `
    -sm hybrid
    #-p "Once upon a time, a girl named Alice" `
