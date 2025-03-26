# Llama.cpp Serving Demo with Openwebui

## Backend with Llama.cpp Serving

1. Build and install llama.cpp from source code

   1. install oneapi 2025.1 from this [link](https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2025-0/apt-005.html), and source vars.

   ```bash
    source /home/intel/oneapi/setvars.sh
   ```

   2. build llama backend

   ```bash
   git clone https://github.com/intel-analytics/llm.cpp.git
   cd llm.cpp/bigdl-core-xe/llama_backend
   mkdir build
   cd build
   cmake .. -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
   cmake --build . -j
   sudo cp libllama_bigdl_core.a /usr/lib/
   ```

   3. build llama.cpp

   ```bash
    git clone https://github.com/intel-analytics/llama-cpp-bigdl.git
    cd llama-cpp-bigdl
    mkdir build
    cd build
    cmake .. -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DLLAMA_BUILD_TESTS=OFF -DGGML_USE_BIGDL=ON
    cmake --build . --config Release -j -v
    cd bin
   ```

   4. start DeepSeek-Q4_K_M using llama.cpp in the `llama-cpp-bigdl/build/bin` directory
      - t: thread, recomment to the physic core in the machine
      - c: ctx too large may lead to OOM of GPU, the biggest test is 10240
      - alias: the model name serving
      - port: the port of llamacpp serving

   * w9:
   ```bash
    export SYCL_CACHE_PERSISTENT=1
    export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
    export IPEX_LLM_SCHED_MAX_COPIES=1
    export IPEX_LLM_QUANTIZE_KV_CACHE=1 # to enable fp8 kv cache
    /home/intel/junwang/llama-cpp-bigdl/build/bin/llama-server -m /home/intel/LLM/DeepSeek-R1-GGUF/DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf -t 60 -e -ngl 99 -c 10240 --temp 0 --no-context-shift -ot exps=CPU --no-mmap --host 0.0.0.0 --port 8001 --alias DeepSeek-R1-Q4_K_M
   ```
   * 云尖:
   ```bash
    export SYCL_CACHE_PERSISTENT=1
    export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
    export IPEX_LLM_SCHED_MAX_COPIES=1
    export IPEX_LLM_QUANTIZE_KV_CACHE=1 # to enable fp8 kv cache
    /home/intel/junwang/llama-cpp-bigdl/build/bin/llama-server -m /home/intel/LLM/DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf -t 64 -e -ngl 99 -c 10240 --no-context-shift -ot exps=CPU --no-mmap --host 0.0.0.0 --port 8001 --alias DeepSeek-R1-Q4_K_M 
   ```

## Frontend with Openwebui

### openwebui

1. Start docker using `openwebui-520.sh` to start openwebui version 5.20

```bash
bash openwebui-520.sh
```

2. Visit <http://localhost:3000> sign up or sign in

   - username: `bigdl@intel.com`
   - password: `intel123`

3. Since llamacpp's serve has serious performance loss in concurrent situations, it is necessary to disable the three functions of openwebui that cause concurrency:

   1. visit <http://localhost:3000/admin/settings>
   2. click the `Inferface` button on left bar
   3. disable `Title Generation`, `Tags Generation` and `Autocomplete Generation` and click `save` to make the config work
      ![](assets\readme_2025-03-26_09-58-07.png)

4. Add llamacpp serving api to Openwebui
   1. visit <http://localhost:3000/admin/settings>
   2. click the `Connection` button on left bar
   3. recommend to disable the `Ollama API` to reduce to useless request send to ollama serving address
   4. click `+` button on upper right, and `url` is the llama.cpp address `http://localhost:8001/v1`, `api-key` can be any string, click the sync button to check the api address is available, and save it.
   ![](assets\readme_2025-03-26_11-26-00.png)
