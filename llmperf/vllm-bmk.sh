export no_proxy=localhost,127.0.0.1
export OPENAI_API_KEY="123456"
export OPENAI_API_BASE="http://localhost:8000/v1"

python token_benchmark_ray.py \
  --model "Qwen2.5-7B-Instruct" \
  --mean-input-tokens 1024 \
  --mean-output-tokens 512 \
  --stddev-input-tokens 10 \
  --stddev-output-tokens 10 \
  --max-num-completed-requests 4 \
  --timeout 600 \
  --num-concurrent-requests 1 \
  --llm-api openai \
  --results-dir "result_outputs" \
  --additional-sampling-params '{}'
