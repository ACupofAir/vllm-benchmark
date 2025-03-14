export DOCKER_IMAGE=intelanalytics/ipex-llm-serving-xpu:2.2.0-b13
export CONTAINER_NAME=junwang-llm-b13

docker rm -f $CONTAINER_NAME
sudo docker run -itd \
        --net=host \
        --group-add video \
        --device=/dev/dri \
	-v /home/arda/LLM:/llm/models/ \
	-v /home/arda/junwang/vllm-benchmark:/llm/workspace \
        -e no_proxy=localhost,127.0.0.1 \
	-e http_proxy=http://proxy.ims.intel.com:911 \
	-e https_proxy=http://proxy.ims.intel.com:911 \
        --memory="32G" \
        --name=$CONTAINER_NAME \
        --shm-size="16g" \
        $DOCKER_IMAGE
