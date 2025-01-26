set -x
# model_name=$1
curl http://localhost:8001/v1/completions -H "Content-Type: application/json" -d '{"model": Qwen2.5-14B-Instruct, "prompt": "San Francisco is a", "max_tokens": 128 }'
