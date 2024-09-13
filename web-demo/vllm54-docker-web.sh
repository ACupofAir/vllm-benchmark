export DOCKER_IMAGE=intelanalytics/ipex-llm-serving-xpu:latest
export CONTAINER_NAME=junwang-vllm54-web

docker rm -f $CONTAINER_NAME
sudo docker run -itd \
	--privileged \
	--net=host \
	--device=/dev/dri \
	-v /home/intel/LLM/:/llm/models \
	-v /home/intel/junwang/:/workspace \
	-e http_proxy="http://proxy.iil.intel.com:911/" \
	-e https_proxy="http://proxy.iil.intel.com:911/" \
	-e no_proxy=localhost,127.0.0.1 \
	--name=$CONTAINER_NAME \
	--shm-size="16g" \
	$DOCKER_IMAGE

