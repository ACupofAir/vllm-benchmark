import intel_extension_for_pytorch as ipex
import xe_linear
import torch
from torch import nn
import time
from ipex_llm.transformers.low_bit_linear import (
    FP4Params,
    ggml_tensor_qtype,
)
import tqdm


def bench_matmul_1st_forward(M, K, N, warm_up, iter_num, device, dtype):

    input_feature = M
    hidden_size = K
    out_feature = N
    qtype = "fp8"

    x = torch.randn([input_feature, hidden_size], dtype=dtype, device=device)
    linear = nn.Linear(hidden_size, out_feature)

    weights = FP4Params(
        data=linear.weight.data,
        requires_grad=False,
        quantized=False,
        _shape=None,
        qtype=ggml_tensor_qtype[qtype],
        enable_xetla=False,
        enable_scale_search=False,
    ).to(device)
    total_time = 0
    for i in tqdm.tqdm(range(warm_up + iter_num)):
    # for i in range(warm_up + iter_num):
        x_2d = x.contiguous().view(-1, x.shape[-1]).clone()
        torch.xpu.synchronize()
        st = time.time()
        xe_linear.forward_new(x_2d, weights, ggml_tensor_qtype[qtype], input_feature)
        torch.xpu.synchronize()
        et = time.time()
        if i >= warm_up:
            total_time += (et - st) * 1000

    avg_latency = total_time / iter_num
    tflops = (2 * M * K * N + 3 * M * N) / avg_latency / 1e12 * 1000
    print(
        f"Shape: {M}x{K}:{K}x{N}, Data Type:{dtype}:{qtype}, Avg Latency: {avg_latency:.2f}ms, TFLOPS: {tflops:.2f} "
    )


if __name__ == "__main__":
    print(torch.__config__.show())
    warm_up = 30
    iter_num = 1000
    matrix_sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
    # matrix_sizes = [8192, 16384]
    device = torch.device("xpu")
    dtype = torch.float16

    for size in matrix_sizes:
        M = K = N = size
        bench_matmul_1st_forward(M, K, N, warm_up, iter_num, device, dtype)