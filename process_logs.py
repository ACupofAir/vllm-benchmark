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

def custom_sort_key(row):
    # 自定义排序规则
    filename = row['filename']
    low_bit = row['low_bit']
    
    if filename.startswith('single_card'):
        prefix_order = 0
    elif filename.startswith('dual_card'):
        prefix_order = 1
    else:
        prefix_order = 2  # 如果有其他类型的文件名前缀

    low_bit_order = {'fp16': 0, 'fp8': 1, 'fp8_e4m3': 2, 'fp6': 3, 'sym_int4': 4}
    low_bit_order_val = low_bit_order.get(low_bit, 5)  # 默认其他值排在最后

    return (prefix_order, low_bit_order_val)

if __name__ == "__main__":
    logs_dir = 'logs'
    df = process_logs(logs_dir)
    # 添加排序键列
    df['sort_key'] = df.apply(custom_sort_key, axis=1)
    # 根据排序键列排序
    df = df.sort_values(by='sort_key').drop(columns='sort_key')
    df.to_csv('logs_summary.csv', index=False)
    print("Logs processed and summary saved to logs_summary.csv")
    # 打印表格
    print(df.to_markdown(index=False))

