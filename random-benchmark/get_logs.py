import re
import csv

with open(r'C:\Users\wangjun9\WorkSpace\vllm-benchmark\random-benchmark\logs\Qwen2.5-14B-Instruct-1024-gpu4-results.log', 'r', encoding='utf-8') as file:
    log_data = file.read()

pattern = re.compile(
        r"Running benchmark with num_prompt=(\d+),.*?"
        r"Request throughput \(req/s\):\s+([\d.]+).*?"
        r"Output token throughput \(tok/s\):\s+([\d.]+).*?"
        r"Total Token throughput \(tok/s\):\s+([\d.]+).*?"
        r"Mean TTFT \(ms\):\s+([\d.]+).*?"
        r"Mean TPOT \(ms\):\s+([\d.]+).*?"
        r"Mean ITL \(ms\):\s+([\d.]+)",
        re.DOTALL
)

matches = pattern.findall(log_data)

with open('benchmark_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['num_prompt', 'req/s', 'Output TPS', 'Total TPS', 'TTFT(mean)', 'TPOT(mean)', 'ITL(mean)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for match in matches:
                writer.writerow({
                        'num_prompt': match[0],
                        'req/s': match[1],
                        'Output TPS': match[2],
                        'Total TPS': match[3],
                        'TTFT(mean)': match[4],
                        'TPOT(mean)': match[5],
                        'ITL(mean)': match[6]
                })