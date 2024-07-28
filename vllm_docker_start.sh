export DOCKER_IMAGE=intelanalytics/ipex-llm-serving-vllm-xpu-experiment:latest
export CONTAINER_NAME=junwang-vllm-bmt

docker rm -f $CONTAINER_NAME
sudo docker run -itd \
        --net=host \
        --device=/dev/dri \
        --name=$CONTAINER_NAME \
        -v /home/intel/LLM:/llm/models/ \
        -v /home/intel/junwang/vllm-benchmark:/llm/vllm-benchmark/ \
        -e http_proxy=http://proxy.iil.intel.com:911 \
        -e https_proxy=http://proxy.iil.intel.com:911 \
        --shm-size="16g" \
        $DOCKER_IMAGE

