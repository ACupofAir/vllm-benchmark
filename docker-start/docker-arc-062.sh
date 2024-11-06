#!/bin/bash
export DOCKER_IMAGE=10.239.45.10/arda/intelanalytics/ipex-llm-serving-xpu:test-image-062
#intelanalytics/vllm-ipex-experimental:txytest_1010
export CONTAINER_NAME=junwang-llm-062

docker rm -f $CONTAINER_NAME
sudo docker run -itd \
	--privileged \
	--net=host \
	--device=/dev/dri \
	--name=$CONTAINER_NAME \
	-v /mnt/disk1/models:/llm/models/ \
	-v /home/arda/junwang:/llm/workspace \
	-e no_proxy=localhost,127.0.0.1 \
	-e http_proxy=http://proxy.iil.intel.com:911 \
	-e https_proxy=http://proxy.iil.intel.com:911 \
	--shm-size="16g" \
	$DOCKER_IMAGE
