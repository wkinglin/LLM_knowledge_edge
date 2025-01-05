import json

def clean_duplicate_arrays(file_path, output_path):
    # 读取文件内容
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 存储处理后的数组
    processed_arrays = []
    prev_length = 0
    
    # 遍历每一行
    for line in lines:
        try:
            # 解析当前行的JSON数组
            current_array = json.loads(line.strip())
            
            # 如果当前数组长度大于之前的数组长度
            if len(current_array) > prev_length:
                # 提取新增的部分
                new_elements = current_array[prev_length:]
                if new_elements:  # 如果有新元素
                    processed_arrays.append(new_elements)
                    prev_length = len(current_array)
        except json.JSONDecodeError:
            continue
    
    # 将处理后的内容写回文件
    with open(output_path, 'w') as f:
        for array in processed_arrays:
            f.write(json.dumps(array) + '\n')

# 使用方法
file_path = './MMLU_ID_Qwen1.5B_no_ft.jsonl'
output_path = './MMLU_ID_Qwen1.5B_no_ft_change.jsonl'
clean_duplicate_arrays(file_path, output_path)