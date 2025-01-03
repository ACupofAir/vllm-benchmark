export DOCKER_IMAGE=intelanalytics/ipex-llm-xpu:latest
export CONTAINER_NAME=ipex-lst
export MODEL_PATH=/home/arda/LLM

docker rm -f $CONTAINER_NAME
docker run -itd \
  --net=host \
  --device=/dev/dri \
  --memory="32G" \
  --name=$CONTAINER_NAME \
  --shm-size="32g" \
  -v $MODEL_PATH:/llm/models \
  -v /home/arda/junwang/vllm-benchmark:/llm/workspace \
  $DOCKER_IMAGE
