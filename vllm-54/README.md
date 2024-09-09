# vllm0.5.4 Benchmark

## offline output verify

```bash
python vllm-out-verify.py <model-path> <tp-num>
```

## online serve benchmark

1. start serve
change the <model-path>, config <model-id> and tp/pp setting in `vllm-serve.sh`, then:

```bash
bash vllm-serve.sh
```

2. benchmark(default 1k-512)

```bash
cd /llm
python vllm_online_benchmark.py <model-id> <max-seq>
```
