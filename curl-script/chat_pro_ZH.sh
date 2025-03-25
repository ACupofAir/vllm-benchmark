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
      "content": "你好"
    },
    {
      "role": "assistant",
      "content": "你好，有什么可以帮助你的吗？"
    },
    {
      "role": "user",
      "content": "请帮忙给出下面这道题的解答：一个工厂生产两种产品 A 和 B，生产一件产品 A 需要消耗原材料甲 3 千克、原材料乙 2 千克，耗时 4 小时，可获利 80 元；生产一件产品 B 需要消耗原材料甲 2 千克、原材料乙 4 千克，耗时 3 小时，可获利 100 元。现在工厂有原材料甲 100 千克，原材料乙 120 千克，总生产时间为 150 小时。问：如何安排生产产品 A 和产品 B 的数量，才能使工厂获得最大利润，最大利润是多少？"
    }
  ],
  "max_tokens": 150,
  "temperature": 0.7
}
EOF
)

response=$(curl -s -X POST "${headers[@]}" -d "$data" "$url")
echo "$response"
