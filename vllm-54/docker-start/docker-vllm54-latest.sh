#!/bin/bash
export DOCKER_IMAGE=intelanalytics/ipex-llm-serving-xpu:latest
export CONTAINER_NAME=junwang-ipex-llm-serving-xpu-container

docker rm -f $CONTAINER_NAME
sudo docker run -itd \
	--net=host \
	--device=/dev/dri \
	--name=$CONTAINER_NAME \
	-v /home/intel/LLM:/llm/models/ \
    -v /home/intel/junwang:/workspace \
	-e no_proxy=localhost,127.0.0.1 \
	--shm-size="16g" \
	$DOCKER_IMAGE

