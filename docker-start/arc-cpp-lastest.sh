#!/bin/bash
export DOCKER_IMAGE=intelanalytics/ipex-llm-inference-cpp-xpu:latest
export CONTAINER_NAME=junwang-cpp-lst

docker rm -f $CONTAINER_NAME
sudo docker run -itd \
	--privileged \
	--net=host \
	--device=/dev/dri \
	--name=$CONTAINER_NAME \
	-v /mnt/disk1/models:/llm/models/ \
	-v /home/arda/junwang/vllm-benchmark:/llm/workspace \
	-e no_proxy=localhost,127.0.0.1 \
	-e http_proxy=http://proxy.iil.intel.com:911 \
	-e https_proxy=http://proxy.iil.intel.com:911 \
	--shm-size="16g" \
	$DOCKER_IMAGE
