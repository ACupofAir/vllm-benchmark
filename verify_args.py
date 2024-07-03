import os
import re
import pandas as pd

def extract_config_info(script_content):
    # 使用正则表达式提取所需的信息
    model = re.search(r'export MODEL="([^"]+)"', script_content).group(1)
    gpu_utilization_rate = re.search(r'export GPU_UTILIZATION_RATE=([\d\.]+)', script_content).group(1)
    tensor_parallel_size = re.search(r'export TENSOR_PARALLEL_SIZE=(\d+)', script_content).group(1)
    max_num_seqs = re.search(r'export MAX_NUM_SEQS=(\d+)', script_content).group(1)
    num_prompts = re.search(r'export NUM_PROMPTS=(\d+)', script_content).group(1)
    max_num_batched_tokens = re.search(r'export MAX_NUM_BATHCED_TOKENS=(\d+)', script_content).group(1)
    max_model_len = re.search(r'export MAX_MODEL_LEN=(\d+)', script_content).group(1)
    low_bit = re.search(r'export LOW_BIT="([^"]+)"', script_content).group(1)

    return {
        'model': model,
        'gpu_utilization_rate': gpu_utilization_rate,
        'tensor_parallel_size': tensor_parallel_size,
        'max_num_seqs': max_num_seqs,
        'num_prompts': num_prompts,
        'max_num_batched_tokens': max_num_batched_tokens,
        'max_model_len': max_model_len,
        'low_bit': low_bit
    }


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


def process_scripts(scripts_dirs):
    data = []
    for scripts_dir in scripts_dirs:
        if os.path.isdir(scripts_dir):
            for filename in os.listdir(scripts_dir):
                if filename.endswith('.sh'):
                    file_path = os.path.join(scripts_dir, filename)
                    with open(file_path, 'r') as file:
                        script_content = file.read()
                        info = extract_config_info(script_content)
                        info['filename'] = f"{scripts_dir}/{filename}"
                        data.append(info)
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    scripts_dirs = ["dual_cards/llama2-7b/", "single_card/llama2-7b/"]  # 替换为你的脚本所在目录
    df = process_scripts(scripts_dirs)

    # 调整列的顺序，将 filename 放在最前
    cols_order = [
        'filename', 'model', 'low_bit', 'gpu_utilization_rate', 'tensor_parallel_size',
        'max_num_seqs', 'num_prompts', 'max_num_batched_tokens', 'max_model_len'
    ]
    df = df[cols_order]

    df['sort_key'] = df.apply(custom_sort_key, axis=1)

    df = df.sort_values(by='sort_key').drop(columns='sort_key')

    # 自定义排序顺序
    # low_bit_order = ['fp16', 'fp8', 'fp8_e4m3', 'fp6', 'sym_int4']
    # df['low_bit'] = pd.Categorical(df['low_bit'], categories=low_bit_order, ordered=True)
    # df = df.sort_values(by='low_bit')
    # 打印生成的表格
    print(df.to_markdown(index=False))

