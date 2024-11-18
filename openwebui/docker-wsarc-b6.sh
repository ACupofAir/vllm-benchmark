#!/bin/bash
export DOCKER_IMAGE=intelanalytics/ipex-llm-serving-xpu:2.2.0-b6
export CONTAINER_NAME=junwang-llm-b6

docker rm -f $CONTAINER_NAME
sudo docker run -itd \
	--privileged \
	--net=host \
	--device=/dev/dri \
	--name=$CONTAINER_NAME \
	-v /data/LLM:/llm/models/ \
	-v /root/junwang/vllm-benchmark:/llm/workspace \
	-e no_proxy=localhost,127.0.0.1 \
	-e http_proxy=http://proxy.iil.intel.com:911 \
	-e https_proxy=http://proxy.iil.intel.com:911 \
	--shm-size="16g" \
	$DOCKER_IMAGE
