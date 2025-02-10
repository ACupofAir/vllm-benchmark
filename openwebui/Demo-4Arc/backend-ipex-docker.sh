#!/bin/bash
export DOCKER_IMAGE=intelanalytics/ipex-llm-serving-xpu:2.2.0-b12
export CONTAINER_NAME=ipex-llm-b12

docker rm -f $CONTAINER_NAME
sudo docker run -itd \
	--privileged \
	--net=host \
	--device=/dev/dri \
	--name=$CONTAINER_NAME \
	-v /data:/llm/models/ \
	-v /home/intel/Demo-4Arc:/llm/workspace \
	-e no_proxy=localhost,127.0.0.1 \
	-e http_proxy=http://proxy.iil.intel.com:911 \
	-e https_proxy=http://proxy.iil.intel.com:911 \
	--shm-size="32g" \
	$DOCKER_IMAGE
