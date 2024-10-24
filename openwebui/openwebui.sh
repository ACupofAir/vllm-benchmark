#!/bin/bash
export DOCKER_IMAGE=ghcr.io/open-webui/open-webui:main
export CONTAINER_NAME=junwang-open-webui

docker rm -f $CONTAINER_NAME

docker run -itd \
           --net=host \
           -p 3000:8080 \
           -e OPENAI_API_KEY=NULL \
           -e OPENAI_API_BASE_URL=http://localhost:8001/v1 \
           -v open-webui:/app/backend/data \
           --name $CONTAINER_NAME \
           --restart always $DOCKER_IMAGE
