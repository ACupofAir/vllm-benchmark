#!/bin/bash
export DOCKER_IMAGE=ghcr.io/open-webui/open-webui:v0.5.20
export CONTAINER_NAME=junwang-open-webui-520

docker rm -f $CONTAINER_NAME

docker run -itd \
  -p 3000:8080 \
  -e http_proxy=http://proxy.iil.intel.com:911 \
  -e https_proxy=http://proxy.iil.intel.com:911 \
  -e no_proxy=localhost,127.0.0.1,10.112.228.157,10.239.45.97,172.16.182.116,10.240.98.152 \
  -e AIOHTTP_CLIENT_TIMEOUT=10 \
  -e AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST=10 \
  -e AIOHTTP_CLIENT_TIMEOUT_OPENAI_MODEL_LIST=10 \
  -e HF_HUB_OFFLINE=1 \
  -v open-webui:/app/backend/data \
  --name $CONTAINER_NAME \
  --restart always $DOCKER_IMAGE
