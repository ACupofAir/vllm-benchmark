set -x
model_name=$1
port=${2:-"8001"}
curl http://localhost:$port/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer intel123" \
  -d "{
  \"model\": \"$model_name\",
  \"prompt\": \"Hello<｜Assistant｜>\",
  \"max_tokens\": 50
}"
