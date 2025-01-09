import ipex_llm
import torch
import torch.nn as nn
import time
import tqdm


def bench_linear(M, K, N, warm_up, iter_num, device, dtype):
    input_tensor = torch.randn(M, K, device=device, dtype=dtype)
    linear = nn.Linear(
        in_features=K, out_features=N, bias=False, device=device, dtype=dtype
    )
    with torch.no_grad():
        linear.weight.copy_(torch.randn(N, K))

    total_time = 0
    for i in tqdm.tqdm(range(warm_up + iter_num)):
        torch.xpu.synchronize()
        st = time.time()
        output = linear(input_tensor)
        torch.xpu.synchronize()
        et = time.time()
        ###
        if i >= warm_up:
            total_time += (et - st) * 1000

    avg_latency = total_time / iter_num
    tflops = (2 * M * K * N + 3 * M * N) / avg_latency / 1e12 * 1000
    print(
        f"Shape: {M}x{K}:{K}x{N}, Data Type:{dtype}, TFLOPS: {tflops:.2f}, Avg Latency: {avg_latency:.2f}"
    )


if __name__ == "__main__":

    print(torch.__config__.show())
    matrix_sizes = [
        (3000, 3584, 4608),
        (3000, 3584, 3584),
        (3000, 3584, 18944),
        (256, 3584, 4096),
        (256, 3584, 512),
        (2048, 3584, 4608),
        (2048, 3584, 3584),
        (2048, 3584, 37888),
        (2048, 18944, 3584),
    ]
    device = torch.device("xpu")
    dtype = torch.float16  # * 87.91 tflops
    # dtype = torch.float32 # * 16.85 tflops

    for size in matrix_sizes:
        M, K, N = size
        bench_linear(M, K, N, 30, 1000, device, dtype)
