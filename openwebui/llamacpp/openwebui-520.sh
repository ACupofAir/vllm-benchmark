#!/bin/bash
export DOCKER_IMAGE=ghcr.io/open-webui/open-webui:v0.5.20
export CONTAINER_NAME=demo-open-webui-520

docker rm -f $CONTAINER_NAME

docker run -itd \
  -p 3000:8080 \
  -e AIOHTTP_CLIENT_TIMEOUT_MODEL_LIST=10 \
  -e AIOHTTP_CLIENT_TIMEOUT_OPENAI_MODEL_LIST=10 \
  -e ENABLE_AUTOCOMPLETE_GENERATION=False \
  -e HF_HUB_OFFLINE=1 \
  -v open-webui:/app/backend/data \
  --name $CONTAINER_NAME \
  --restart always $DOCKER_IMAGE
  #-e http_proxy=http://proxy.iil.intel.com:911 \
  #-e https_proxy=http://proxy.iil.intel.com:911 \
  #-e no_proxy=localhost,127.0.0.1,1
