#!/bin/bash

# 定义要测试的 num_prompt 和 random-input-len 值
# num_prompt_values=(2 1 2 4 6 8 10 12 14 16 18 20 22 24)
num_prompt_values=(2 1 1)
random_input_len=1024
random_output_len=128
model_name="Llama-3.1-8B-Instruct"
gpu_num=1
port=8001

# 基础命令
base_command="python /llm/vllm/benchmarks/benchmark_serving.py \
	      --model /llm/models/${model_name} \
	      --dataset-name random \
	      --trust_remote_code \
	      --port ${port}"

log_file="int8-${model_name}-${random_input_len}-${random_output_len}-gpu${gpu_num}-g95-2000-3000.log"

# 清空或创建日志文件
>"$log_file"

for num_prompt in "${num_prompt_values[@]}"; do
  echo "Running benchmark with batch size ${num_prompt}, random-input-len=${random_input_len}..." | tee -a "$log_file"
  $base_command --num_prompt "$num_prompt" --random-input-len "$random_input_len" --random-output-len "$random_output_len" >> "$log_file" 2>&1
  echo "Completed benchmark with batch size ${num_prompt}, random-input-len=${random_input_len}." | tee -a "$log_file"
  echo "------------------------------------------------------------" | tee -a "$log_file"
done

echo "All benchmarks completed. Results are saved in $log_file"

python get_logs.py $log_file
