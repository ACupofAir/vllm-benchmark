import re
import pandas as pd

# input =
"""input:
Running benchmark with batch size 1
max_seq: 1
input_length: 1024
output_length: 512
Warm Up:   0%|                                                               | 0/2 [00:00<?, ?req/s]
Warm Up:  50%|███████████████████████████▌                           | 1/2 [00:27<00:27, 27.34s/req]
Warm Up: 100%|███████████████████████████████████████████████████████| 2/2 [00:29<00:00, 12.71s/req]
Warm Up: 100%|███████████████████████████████████████████████████████| 2/2 [00:29<00:00, 14.90s/req]
Warm Up:   0%|                                                               | 0/2 [00:00<?, ?req/s]
Warm Up:  50%|███████████████████████████▌                           | 1/2 [00:31<00:31, 31.65s/req]
Warm Up: 100%|███████████████████████████████████████████████████████| 2/2 [01:03<00:00, 31.64s/req]
Warm Up: 100%|███████████████████████████████████████████████████████| 2/2 [01:03<00:00, 31.65s/req]
Benchmarking:   0%|                                                          | 0/4 [00:00<?, ?req/s]
Benchmarking:  25%|████████████▌                                     | 1/4 [00:31<01:34, 31.60s/req]
Benchmarking:  50%|█████████████████████████                         | 2/4 [01:03<01:03, 31.60s/req]
Benchmarking:  75%|█████████████████████████████████████▌            | 3/4 [01:34<00:31, 31.63s/req]
Benchmarking: 100%|██████████████████████████████████████████████████| 4/4 [02:06<00:00, 31.62s/req]
Benchmarking: 100%|██████████████████████████████████████████████████| 4/4 [02:06<00:00, 31.62s/req]
Total time for 4 requests with 1 concurrent requests: 126.47350115102017 seconds.
Average responce time: 31.618342171263066
Token throughput: 16.1931154064796
/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py:3504: RuntimeWarning: Mean of empty slice.
  return _methods._mean(a, axis=axis, dtype=dtype,
/usr/local/lib/python3.11/dist-packages/numpy/core/_methods.py:129: RuntimeWarning: invalid value encountered in scalar divide
  ret = ret.dtype.type(ret / rcount)
Average first token latency: 612.7748937433353 milliseconds.
P90 first token latency: 613.407070923131 milliseconds.
P95 first token latency: 613.4372774686199 milliseconds.
Average next token latency: 60.67521886032089 milliseconds.
P90 next token latency: 60.73026275594363 milliseconds.
P95 next token latency: 60.74391092213131 milliseconds.
Running benchmark with batch size 2
running bench.py
model_name: Qwen1.5-32B-Chat
max_seq: 2
input_length: 1024
output_length: 512
Warm Up:   0%|                                                               | 0/2 [00:00<?, ?req/s]
Warm Up:  50%|███████████████████████████▌                           | 1/2 [00:02<00:02,  2.45s/req]
Warm Up: 100%|███████████████████████████████████████████████████████| 2/2 [00:04<00:00,  2.46s/req]
Warm Up: 100%|███████████████████████████████████████████████████████| 2/2 [00:04<00:00,  2.46s/req]
Warm Up:   0%|                                                               | 0/4 [00:00<?, ?req/s]
Warm Up:  25%|█████████████▊                                         | 1/4 [00:37<01:51, 37.27s/req]
Warm Up:  75%|█████████████████████████████████████████▎             | 3/4 [01:10<00:21, 21.79s/req]
Warm Up: 100%|███████████████████████████████████████████████████████| 4/4 [01:10<00:00, 17.50s/req]
Benchmarking:   0%|                                                          | 0/8 [00:00<?, ?req/s]
Benchmarking:  12%|██████▎                                           | 1/8 [00:32<03:47, 32.52s/req]
Benchmarking:  38%|██████████████████▊                               | 3/8 [01:05<01:42, 20.52s/req]
Benchmarking:  62%|███████████████████████████████▎                  | 5/8 [01:37<00:55, 18.36s/req]
Benchmarking:  88%|███████████████████████████████████████████▊      | 7/8 [02:10<00:17, 17.50s/req]
Total time for 8 requests with 2 concurrent requests: 130.43152660201304 seconds.
Average responce time: 32.6078159812605
Benchmarking: 100%|██████████████████████████████████████████████████| 8/8 [02:10<00:00, 16.30s/req]
Token throughput: 31.403450582144636
/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py:3504: RuntimeWarning: Mean of empty slice.
  return _methods._mean(a, axis=axis, dtype=dtype,
/usr/local/lib/python3.11/dist-packages/numpy/core/_methods.py:129: RuntimeWarning: invalid value encountered in scalar divide
  ret = ret.dtype.type(ret / rcount)
Average first token latency: 967.7003930119099 milliseconds.
P90 first token latency: 1220.8141462877393 milliseconds.
P95 first token latency: 1221.2209411431104 milliseconds.
Average next token latency: 61.91787907668829 milliseconds.
P90 next token latency: 62.6864475155062 milliseconds.
P95 next token latency: 62.68704466559974 milliseconds.
"""


def extract_benchmark_data(input_text, output_csv, columns=None):
    pattern = re.compile(
        r"batch size (\d+).*?Average responce time: ([\d.]+).*?Token throughput: ([\d.]+).*?Average first token latency: ([\d.]+).*?P90 first token latency: ([\d.]+).*?P95 first token latency: ([\d.]+).*?Average next token latency: ([\d.]+).*?P90 next token latency: ([\d.]+).*?P95 next token latency: ([\d.]+)",
        re.DOTALL,
    )

    data = []
    matched = pattern.finditer(input_text)
    for match in matched:
        (
            batch_size,
            avg_resp_time,
            token_throughput,
            avg_first_token_latency,
            p90_first_token_latency,
            p95_first_token_latency,
            avg_next_token_latency,
            p90_next_token_latency,
            p95_next_token_latency,
        ) = match.groups()
        data.append(
            [
                batch_size,
                avg_resp_time,
                token_throughput,
                avg_first_token_latency,
                p90_first_token_latency,
                p95_first_token_latency,
                avg_next_token_latency,
                p90_next_token_latency,
                p95_next_token_latency,
            ]
        )

    df = pd.DataFrame(
        data,
        columns=[
            "Batch Size",
            "Average Response Time",
            "Token Throughput",
            "Average First Token Latency",
            "P90 First Token Latency",
            "P95 First Token Latency",
            "Average Next Token Latency",
            "P90 Next Token Latency",
            "P95 Next Token Latency",
        ],
    )

    if columns:
        df = df[columns]

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    print(df)
    df.to_excel(output_csv, index=False)


# Example usage
if __name__ == "__main__":
    input_file = (
        r"C:\Users\wangjun9\WorkSpace\vllm-benchmark\other-scripts\tmp_input.txt"
    )
    output_file = "qwen1.5-32b-b3.xlsx"
    with open(input_file, "r", encoding="utf-8") as file:
        input = file.read()

    extract_benchmark_data(
        input,
        output_file,
        columns=[
            "Batch Size",
            "Average Response Time",
            "Token Throughput",
            "Average First Token Latency",
            "P90 First Token Latency",
            "P95 First Token Latency",
            "Average Next Token Latency",
            "P90 Next Token Latency",
            "P95 Next Token Latency",
        ],
    )
