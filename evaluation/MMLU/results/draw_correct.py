import matplotlib.pyplot as plt
import numpy as np

# 数据
methods = ['Origin', 'Fine-Tuning', 'Fine-Tuning Sample']
correct = [0.5850, 0.2050, 0.3922]
refuse = [0, 0.5892, 0.3399]
wrong = [0.4150, 0.2058, 0.2679]

# 设置柱的位置
x = np.arange(len(methods))

# 设置柱的宽度
width = 0.5

# 创建一个绘图
fig, ax = plt.subplots()

# 绘制堆叠柱状图
ax.bar(x, correct, width, label='Correct', color='green')
ax.bar(x, refuse, width, bottom=correct, label='Refuse', color='orange')
ax.bar(x, wrong, width, bottom=np.array(correct) + np.array(refuse), label='Wrong', color='red')

# 添加标签、标题等
ax.set_xlabel('Methods')
ax.set_ylabel('Values')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()

plt.savefig('./picture/ID_compare.png', dpi=300, bbox_inches='tight')

