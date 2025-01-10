import json
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
with open('total.json', 'r') as f:
    data = json.load(f)

# 删除total键值对
del data['total']

# 创建数据框
df = pd.DataFrame([
    {
        'category': k,
        'correct_rate': v['right_num'] / v['total_num'],
        'wrong_rate': 1 - (v['right_num'] / v['total_num'])
    }
    for k, v in data.items()
])

# 排序（按正确率降序）
# df = df.sort_values('correct_rate', ascending=False)

# 筛选准确率低于0.4的数据
df = df[df['correct_rate'] < 0.4]

# 创建图形
plt.figure(figsize=(15, 8))

# 绘制堆叠直方图
plt.bar(df['category'], df['correct_rate'], label='Correct', color='#2ecc71')
plt.bar(df['category'], df['wrong_rate'], bottom=df['correct_rate'], label='Wrong', color='#e74c3c')

# 旋转x轴标签以防重叠
plt.xticks(rotation=45, ha='right')

# 添加图例
plt.legend(loc='upper right')

# 添加网格线
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 调整布局，确保标签不被切掉
plt.tight_layout()

plt.savefig('./normalized_distribution.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()