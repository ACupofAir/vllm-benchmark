#!/bin/bash
export DOCKER_IMAGE=registry.cn-chengdu.aliyuncs.com/yangxianpku/ubuntu22.04-cu12.4-torch2.4.0-vllm0.6.3:py3.10-24.10
export CONTAINER_NAME=dianxin

docker rm -f $CONTAINER_NAME
sudo docker run -itd --net=host --name=$CONTAINER_NAME -e no_proxy=localhost,127.0.0.1 $DOCKER_IMAGE bash
