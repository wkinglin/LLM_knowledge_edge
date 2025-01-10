import json

def calculate_sure_ratios(data):
    # 初始化计数器
    total_right_sure = 0
    total_false_sure = 0
    total_questions = 0
    
    # 遍历所有科目（跳过'total'键）
    for subject, stats in data.items():
        if subject != 'total':
            total_right_sure += stats['right_sure']
            total_false_sure += stats['false_sure']
            total_questions += stats['total_num']
    
    # 计算比例
    right_sure_ratio = total_right_sure / total_questions
    false_sure_ratio = total_false_sure / total_questions
    
    return {
        'total_questions': total_questions,
        'total_right_sure': total_right_sure,
        'total_false_sure': total_false_sure,
        'right_sure_ratio': right_sure_ratio,
        'false_sure_ratio': false_sure_ratio
    }

# 读取JSON文件
with open('total.json', 'r') as f:
    data = json.load(f)

# 计算比例
results = calculate_sure_ratios(data)

# 打印结果
print(f"总问题数: {results['total_questions']}")
print(f"确定正确数(right_sure): {results['total_right_sure']}")
print(f"确定错误数(false_sure): {results['total_false_sure']}")
print(f"确定正确比例: {results['right_sure_ratio']:.4f}")
print(f"确定错误比例: {results['false_sure_ratio']:.4f}")