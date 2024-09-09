### prepare docker container

sudo docker pull intelanalytics/ipex-llm-serving-xpu:latest

export DOCKER_IMAGE=intelanalytics/ipex-llm-serving-xpu:latest
 
sudo docker run -itd \
        --net=host \
        --device=/dev/dri \
        --name=demo-container \
        -v /LLM_MODELS/:/llm/models/ \
        --shm-size="16g" \
        $DOCKER_IMAGE
 
sudo docker exec -it demo-container bash
 
 
### start vllm demo
 
vi start-demo.sh
```
export no_proxy="127.0.0.1,localhost"
ps -ef | grep "fastchat" | awk '{print $2}' | xargs kill -9

# start controller
python -m fastchat.serve.controller &
 

#start vllm_worker
export TORCH_LLM_ALLREDUCE=0
export CCL_DG2_ALLREDUCE=1
# CCL needed environment variables
export CCL_WORKER_COUNT=2
# pin ccl worker to cores
# export CCL_WORKER_AFFINITY=32,33,34,35
export FI_PROVIDER=shm
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=sockets
export CCL_ATL_SHM=1
 
source /opt/intel/1ccl-wks/setvars.sh

 
python -m ipex_llm.serving.fastchat.vllm_worker \
--model-path /llm/models/Qwen1.5-32B-Chat \
--device xpu \
--enforce-eager \
--dtype float16 \
--load-in-low-bit fp8 \
--tensor-parallel-size 4 \
--gpu-memory-utilization 0.60 \
--max-model-len 8000 \
--max-num-batched-tokens 8000 &
```

 
vi start-gradio.sh
```
#!/bin/bash
 
# start gradio wed server
python -m fastchat.serve.gradio_web_server
```

Visit 127.0.0.1:7860 in browser to chat.