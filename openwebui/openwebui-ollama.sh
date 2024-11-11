#!/bin/bash
export DOCKER_IMAGE=ghcr.io/open-webui/open-webui:main
export CONTAINER_NAME=junwang-open-webui

docker rm -f $CONTAINER_NAME

docker run -itd \
           -v open-webui:/app/backend/data \
           -e PORT=8081 \
           --privileged \
           --network=host \
           --add-host=host.docker.internal:host-gateway \
           --name $CONTAINER_NAME \
           --restart always $DOCKER_IMAGE
