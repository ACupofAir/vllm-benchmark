#!/bin/bash
export DOCKER_IMAGE=ghcr.io/open-webui/open-webui:main
export CONTAINER_NAME=junwang-open-webui

docker rm -f $CONTAINER_NAME

docker run -itd \
           -p 3000:8080 \
           -e OPENAI_API_BASE_URL=http://10.112.228.104:8001/v1 \
           -e OPENAI_API_KEY=intel123 \
           -v open-webui:/app/backend/data \
           --name $CONTAINER_NAME \
           --restart always $DOCKER_IMAGE
