import os
import re
import pandas as pd

def extract_info(log_content):
    # 使用正则表达式提取所需的信息
    low_bit = re.search(r'export LOW_BIT="(\w+)"', log_content).group(1)
    max_num_batched_tokens = re.search(r'export MAX_NUM_BATHCED_TOKENS=(\d+)', log_content).group(1)
    num_prompts = re.search(r'export NUM_PROMPTS=(\d+)', log_content).group(1)
    max_num_seqs = re.search(r'export MAX_NUM_SEQS=(\d+)', log_content).group(1)
    gpu_utilization_rate = re.search(r'export GPU_UTILIZATION_RATE=([\d\.]+)', log_content).group(1)
    gpu_blocks = re.search(r'INFO.*gpu_executor\.py.*# GPU blocks: (\d+)', log_content).group(1)
    in_len = re.search(r'export IN_LEN=(\d+)', log_content).group(1)
    out_len = re.search(r'export OUT_LEN=(\d+)', log_content).group(1)

    # 计算 expected_max_num_seqs，向下整除
    expected_max_num_seqs = int(gpu_blocks) * 16 // (int(in_len) + int(out_len))

    # 检查 Throughput 信息
    throughput_match = re.search(r'Throughput: ([\d\.]+) requests/s, (\d+\.?\d*) tokens/s', log_content)
    if throughput_match:
        request_per_second = throughput_match.group(1)
        token_throughput = throughput_match.group(2)
    else:
        request_per_second = 'failed'
        token_throughput = 'failed'

    # 提取 total_num_cumulative_preemption 的最新出现值
    preemption_matches = re.findall(r'total_num_cumulative_preemption=(\d+)', log_content)
    if preemption_matches:
        total_num_cumulative_preemption = preemption_matches[-1]
    else:
        total_num_cumulative_preemption = '0'
    
    return {
        'filename': log_content,
        'low_bit': low_bit,
        'max_num_batched_tokens': max_num_batched_tokens,
        'num_prompts': num_prompts,
        'max_num_seqs': max_num_seqs,
        'gpu_utilization_rate': gpu_utilization_rate,
        'gpu_blocks': gpu_blocks,
        'request_per_second': request_per_second,
        'token_throughput': token_throughput,
        'expected_max_num_seqs': expected_max_num_seqs,
        'total_num_cumulative_preemption': total_num_cumulative_preemption
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

    # 调整列的顺序，将 expected_max_num_seqs 放在 max_num_seqs 右边
    cols = df.columns.tolist()
    max_num_seqs_index = cols.index('max_num_seqs')
    cols.insert(max_num_seqs_index + 1, cols.pop(cols.index('expected_max_num_seqs')))
    df = df[cols]
    print(df.to_markdown(index=False))
    print("###########################################")

    # 创建与Excel表格匹配的新DataFrame
    output_data = []
    for _, row in df.iterrows():
        if row['request_per_second'] == 'failed':
            output_token_per_sec = 'failed'
        else:
            output_token_per_sec = float(row['request_per_second']) * 512
        
        output_row = {
            'filename': row['filename'],
            'req per sec': row['request_per_second'],
            'output token per sec (TPS)': output_token_per_sec,
            '1st': '',  # 忽略
            'next': '',  # 忽略
            'owner': '',  # 忽略
            'ratio': '',  # 忽略
            'GPU blocks': row['gpu_blocks'],
            'GPU utilizaton rate': row['gpu_utilization_rate'],
            'max num seqs': row['max_num_seqs'],
            'max num batched tokens': row['max_num_batched_tokens'],
            'in-out throughput': row['token_throughput']
        }
        output_data.append(output_row)
    
    output_df = pd.DataFrame(output_data)

    # 打印生成的表格
    print(output_df.to_markdown(index=False))

    # 保存到csv文件
    output_df.to_csv('logs_summary_for_excel.csv', index=False)
    print("Logs processed and summary saved to logs_summary_for_excel.csv")
