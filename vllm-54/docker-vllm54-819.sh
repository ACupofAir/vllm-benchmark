#!/bin/bash
export DOCKER_IMAGE=10.239.45.10/arda/intelanalytics/vllm-ipex-054:20240812
export CONTAINER_NAME=junwang-vllm54-bmt

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