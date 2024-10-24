import re
import pandas as pd


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
