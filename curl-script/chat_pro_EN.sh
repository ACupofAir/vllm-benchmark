#!/bin/bash
#set -x
if [ -z "$1" ]; then
  echo "Error: need model name"
  echo "Usage: ./request.sh <model_name>"
  exit 1
fi

url="http://localhost:8001/v1/chat/completions"
model_name="$1"

# headers
headers=(
  -H "Content-Type: application/json"
  -H "Authorization: Bearer intel123"
)

# body
data=$(cat <<EOF
{
  "model": "$model_name",
  "messages": [
    {
      "role": "user",
      "content": "Hello"
    },
    {
      "role": "assistant",
      "content": "Hello, what can I do for you?"
    },
    {
      "role": "user",
      "content": "Please help answer the following question: A factory produces two products A and B. To produce one product A, it needs 3 kg of raw material A and 2 kg of raw material B. It runs for 4 hours and can make a profit of 80 yuan. To produce one product B, it needs 2 kg of raw material A and 4 kg of raw material B. It takes 3 hours and can make a profit of 100 yuan. Now the factory has 100 kg of raw material A and 120 kg of raw material B, and the total production time is 150 hours. Question: How to arrange the production quantity of product A and product B so that the factory can obtain the maximum profit, and what is the maximum profit?"
    }
  ],
  "max_tokens": 150,
  "temperature": 0.7
}
EOF
)

# response
response=$(curl -s -X POST "${headers[@]}" -d "$data" "$url")
echo "$response"
