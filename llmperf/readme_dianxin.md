## vllm-serving
1. start docker and start vllm serving with openai api
```bash
# run in docker
bash tp1pp1-qwen25-7b.sh
```

## conda env setting
1. create conda env with python3.8
```bash
conda create -n llmperf python=3.8
```

2. install rust dependency
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

3. install env
```bash
git clone https://github.com/ray-project/llmperf.git
cd llmperf
pip install -e .
```

4. change `vllm-bmk.sh`'s parmas and run
```bash
bash vllm-bmk.sh
```
