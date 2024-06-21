import os
import re
import pandas as pd

def extract_info(log_content):
    # 使用正则表达式提取所需的信息
    low_bit = re.search(r'export LOW_BIT="(\w+)"', log_content).group(1)
    max_num_batched_tokens = re.search(r'export MAX_NUM_BATHCED_TOKENS=(\d+)', log_content).group(1)
    max_num_seqs = re.search(r'export MAX_NUM_SEQS=(\d+)', log_content).group(1)
    gpu_utilization_rate = re.search(r'export GPU_UTILIZATION_RATE=([\d\.]+)', log_content).group(1)
    gpu_blocks = re.search(r'INFO.*gpu_executor\.py.*# GPU blocks: (\d+)', log_content).group(1)

    # 检查 Throughput 信息
    throughput_match = re.search(r'Throughput: ([\d\.]+) requests/s, (\d+\.?\d*) tokens/s', log_content)
    if throughput_match:
        request_per_second = throughput_match.group(1)
        token_throughput = throughput_match.group(2)
    else:
        request_per_second = 'failed'
        token_throughput = 'failed'
    
    return {
        'filename': log_content,
        'low_bit': low_bit,
        'max_num_batched_tokens': max_num_batched_tokens,
        'max_num_seqs': max_num_seqs,
        'gpu_utilization_rate': gpu_utilization_rate,
        'gpu_blocks': gpu_blocks,
        'request_per_second': request_per_second,
        'token_throughput': token_throughput
    }

def process_logs(logs_dir):
    data = []
    for filename in os.listdir(logs_dir):
        if filename.endswith('.log'):
            with open(os.path.join(logs_dir, filename), 'r') as file:
                log_content = file.read()
                info = extract_info(log_content)
                info['filename'] = filename
                data.append(info)
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    logs_dir = 'logs'
    df = process_logs(logs_dir)
    df.to_csv('logs_summary.csv', index=False)
    print("Logs processed and summary saved to logs_summary.csv")
    # 打印表格
    print(df.to_markdown(index=False))

