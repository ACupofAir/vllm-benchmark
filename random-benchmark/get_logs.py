import re
import csv

input_log = r'C:\Users\wangjun9\WorkSpace\vllm-benchmark\random-benchmark\logs\qwen2.5-32b-4gpu.log'
output_csv = input_log.split('.log')[0] + '.csv'
with open(input_log, 'r', encoding='utf-8') as file:
    log_data = file.read()


pattern = re.compile(
        r"Running benchmark with batch size (\d+).*?"
        r"Request throughput \(req/s\):\s+([\d.]+).*?"
        r"Output token throughput \(tok/s\):\s+([\d.]+).*?"
        r"Total Token throughput \(tok/s\):\s+([\d.]+).*?"
        r"Mean TTFT \(ms\):\s+([\d.]+).*?"
        r"Mean TPOT \(ms\):\s+([\d.]+).*?"
        r"Mean ITL \(ms\):\s+([\d.]+)",
        re.DOTALL
)

matches = pattern.findall(log_data)
print('======================DEBUG START: matches======================')
print(matches)
print('======================DEBUG  END : matches======================')

with open(output_csv, 'w', newline='') as csvfile:
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