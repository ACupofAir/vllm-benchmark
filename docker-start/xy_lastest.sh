export DOCKER_IMAGE=intelanalytics/ipex-llm-serving-xpu:latest
export CONTAINER_NAME=txytest_vllm_054
sudo docker run -itd \
        --privileged \
        --net=host \
        --device=/dev/dri \
        -v /home/intel/LLM:/llm/models \
        -e no_proxy=localhost,127.0.0.1 \
        --name=$CONTAINER_NAME \
        -v /home/intel/xiangyu:/llm/workspace \
        --shm-size="16g" \
        $DOCKER_IMAGE
