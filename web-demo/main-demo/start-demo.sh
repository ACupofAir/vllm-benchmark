ps -ef | grep "fastchat" | awk '{print $2}' | xargs kill -9


pip install -U gradio==4.43.0
pip install trl

# start controller
python -m fastchat.serve.controller &


export TORCH_LLM_ALLREDUCE=0
export CCL_DG2_ALLREDUCE=1
# CCL needed environment variables
export CCL_WORKER_COUNT=4
# pin ccl worker to cores
# export CCL_WORKER_AFFINITY=32,33,34,35
export FI_PROVIDER=shm
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=sockets
export CCL_ATL_SHM=1
# gpu usage:
export ZE_AFFINITY_MASK=0,1

source /opt/intel/1ccl-wks/setvars.sh

python -m ipex_llm.serving.fastchat.vllm_worker \
--model-path /llm/models/Qwen1.5-14B-Chat \
--device xpu \
--enforce-eager \
--dtype float16 \
--load-in-low-bit fp8 \
--tensor-parallel-size 2 \
--gpu-memory-utilization 0.75 \
--max-model-len 2048 \
--max-num-batched-tokens 4096 &

sleep 120

python -m fastchat.serve.gradio_web_server &
