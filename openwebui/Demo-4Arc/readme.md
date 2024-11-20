# IPEX vLLM Serving with openwebui 8xARC770

## Backend with vLLM Serving

1. Start docker using `backend-ipex-docker.sh`
    * change  `-v <model-path>:/llm/models` to your model path
    * change `-v <script-path>:/llm/workspace` to the script file path

```bash
bash backend-ipex-docker.sh
```

2. Start IPEX vLLM Serving using `vllm-serving` in docker container

>Only using tp4pp2 need change the openwebui docker backend code

go into docker container

```bash
docker exec -it ipex-llm-b6 bash
```

and start vllm serving

```bash
bash vllm-qwen2.5-72b-openaikey.sh
```

## Frontedn with openwebui

1. start docker using `frontend-openwebui-docker.sh`
change `OPENAI_API_BASE_URL`'s ip to <host-ip>

```bash
bash frontend-openwebui-docker.sh
```

2. visit <https://host-ip:3000> sign up or sign in
    * username: `bigdl@intel.com`
    * password: `intel123`
