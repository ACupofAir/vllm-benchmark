$model_name=$1
curl http://localhost:8001/v1/completions \
	-H "Content-Type: application/json" \
	-d '{"model": $model_name,
	"prompt": "San Francisco is a",
	"max_tokens": 128
}'
