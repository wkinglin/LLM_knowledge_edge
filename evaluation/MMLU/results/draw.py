import json
import matplotlib.pyplot as plt
import numpy as np

# 读取JSON文件
with open('./MMLU_OOD_Qwen1.5B_overRefuse/total.json', 'r') as f:
    data = json.load(f)

# 删除total键，因为它的结构不同
del data['total']

# 准备数据
categories = list(data.keys())
# 缩短类别名称
shortened_categories = [cat.replace('high_school_', 'hs_').replace('college_', 'col_') for cat in categories]
right_sure = [data[cat]['right_sure']/data[cat]['total_num'] for cat in categories]
right_unsure = [data[cat]['right_unsure']/data[cat]['total_num'] for cat in categories]
false_sure = [data[cat]['false_sure']/data[cat]['total_num'] for cat in categories]
false_unsure = [data[cat]['false_unsure']/data[cat]['total_num'] for cat in categories]

# 设置图形大小 - 增加宽度,减小高度
plt.figure(figsize=(20,10))

# 创建位置数组
x = np.arange(len(categories))
width = 0.8  # 柱的宽度

# 创建堆叠柱状图
plt.bar(x, right_sure, width, label='Correct', color='darkgreen')
plt.bar(x, right_unsure, width, bottom=right_sure, label='Over Refuse', color='lightgreen')
plt.bar(x, false_sure, width, bottom=np.array(right_sure) + np.array(right_unsure), 
        label='Wrong', color='darkred')
plt.bar(x, false_unsure, width, 
        bottom=np.array(right_sure) + np.array(right_unsure) + np.array(false_sure),
        label='Refuse', color='lightcoral')

# 自定义图表
plt.xlabel('Categories')
plt.ylabel('Normalized Proportion')
plt.title('Normalized Distribution of Answer Types by Category')
# 使用缩短的类别名称,减小字体大小
plt.xticks(x, shortened_categories, rotation=45, ha='right', fontsize=8)
plt.legend()

# 调整布局
plt.tight_layout()

plt.savefig('./picture/OOD_overRefuse_modify.png', dpi=300, bbox_inches='tight')

# 显示图表
plt.show()