export DOCKER_IMAGE=10.239.45.10/arda/intelanalytics/ipex-llm-serving-xpu:pytorch26
export CONTAINER_NAME=junwang-llm-torch26

docker rm -f $CONTAINER_NAME
sudo docker run -itd \
	--privileged \
	--net=host \
	--device=/dev/dri \
	--name=$CONTAINER_NAME \
	-v /home/arda/LLM:/llm/models/ \
	-v /home/arda/junwang/vllm-benchmark:/llm/workspace \
	-e no_proxy=localhost,127.0.0.1 \
	-e http_proxy=http://proxy.iil.intel.com:911 \
	-e https_proxy=http://proxy.iil.intel.com:911 \
	--shm-size="32g" \
	$DOCKER_IMAGE""
