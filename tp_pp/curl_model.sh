set -x
model_name=$1
port=$2
curl http://localhost:$port/v1/completions \
  -H "Content-Type: application/json" \
  -d "{
  \"model\": \"$model_name\",
  \"prompt\": \"New York is the \",
  \"max_tokens\": 50
}"
