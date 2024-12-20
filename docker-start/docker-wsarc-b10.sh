#!/bin/bash
export DOCKER_IMAGE=intelanalytics/ipex-llm-serving-xpu:b10-pre
export CONTAINER_NAME=junwang-llm-b10-pre

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
	--shm-size="32g" \
	$DOCKER_IMAGE
