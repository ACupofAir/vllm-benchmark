#!/bin/bash
export DOCKER_IMAGE=intelanalytics/ipex-llm-serving-xpu:2.2.0-b11
export CONTAINER_NAME=junwang-llm-b11

docker rm -f $CONTAINER_NAME
sudo docker run -itd \
  --privileged \
  --net=host \
  --device=/dev/dri \
  --name=$CONTAINER_NAME \
  -v /home/intel/LLM:/llm/models/ \
  -v /home/intel/junwang/vllm-benchmark:/llm/workspace \
  -e no_proxy=localhost,127.0.0.1 \
  -e http_proxy=http://proxy.ims.intel.com:911 \
  -e https_proxy=http://proxy.ims.intel.com:911 \
  --shm-size="16g" \
  $DOCKER_IMAGE
