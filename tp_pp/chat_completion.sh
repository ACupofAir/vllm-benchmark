set -x
# model_name=$1
# port=$2
# curl http://localhost:$port/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer intel123" \
#   -d "{
#   \"model\": \"$model_name\",
#   \"messages\": [
#       {"role": "system", "content": "You are a helpful assistant."},
#       {"role": "user", "content": "What is the capital of France?"}
#   ],
#   \"max_tokens\": 50
# }"

curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer intel123" \
  -d '{
           "model": deepseek-r1:1.5,
           "messages": [
               {"role": "system", "content": "You are a helpful assistant."},
               {"role": "user", "content": "What is the capital of France?"}
           ],
           "max_tokens": 100,
           "temperature": 0.7
         }'
