#!/bin/bash
export DOCKER_IMAGE=ghcr.io/open-webui/open-webui:main
export CONTAINER_NAME=junwang-open-webui-lst

docker rm -f $CONTAINER_NAME

docker run -itd \
           -p 3000:8080 \
           -e http_proxy=http://proxy.iil.intel.com:911 \
           -e https_proxy=http://proxy.iil.intel.com:911 \
           -e no_proxy=localhost,127.0.0.1,10.112.228.157,10.239.45.97,172.16.182.116 \
           -v open-webui:/app/backend/data \
           --name $CONTAINER_NAME \
           --restart always $DOCKER_IMAGE
