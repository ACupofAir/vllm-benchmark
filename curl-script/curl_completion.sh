#!/bin/bash
# usage: ./curl_completion.sh --port xx | jq
set -x

# 默认值
model_name=""
port="8001"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --port)
      port="$2"
      shift
      shift
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

# 发出 curl 请求
curl http://localhost:$port/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer intel123" \
  -d "{
  \"model\": \"$model_name\",
  \"prompt\": \"New York is the \",
  \"max_tokens\": 1,
  \"temperature\": 0.0,
  \"logprobs\": 2,
  \"echo\": \"True\"
}"

