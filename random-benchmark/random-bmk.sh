#!/bin/bash

# 定义要测试的 num_prompt 和 random-input-len 值
# num_prompt_values=(2 1 2 4 6 8 10 12 14 16 18 20 22 24)
num_prompt_values=(20 22 24)
random_input_len=1024
model_name="Qwen2.5-32B-Instruct-GPTQ-Int4"
gpu_num=4
port=8001

# 基础命令
base_command="python /llm/vllm/benchmarks/benchmark_serving.py \
	--model /llm/models/${model_name} \
	--dataset-name random \
	--trust_remote_code \
  --port ${port} \
	--random-output-len=512"

log_file="${model_name}-${random_input_len}-gpu${gpu_num}-g95-2000-3000.log"

# 清空或创建日志文件
>"$log_file"

for num_prompt in "${num_prompt_values[@]}"; do
  echo "Running benchmark with batch size ${num_prompt}, random-input-len=${random_input_len}..." | tee -a "$log_file"
  $base_command --num_prompt "$num_prompt" --random-input-len "$random_input_len" >>"$log_file" 2>&1
  echo "Completed benchmark with batch size ${num_prompt}, random-input-len=${random_input_len}." | tee -a "$log_file"
  echo "------------------------------------------------------------" | tee -a "$log_file"
done

echo "All benchmarks completed. Results are saved in $log_file"

python get_logs.py $log_file
