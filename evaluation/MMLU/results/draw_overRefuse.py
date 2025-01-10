import matplotlib.pyplot as plt
import numpy as np

# 数据
methods = ['Origin', 'Fine-Tuning', 'Fine-Tuning S', 'Fine-tuning L']
correct = [0.5850, 0.2050, 0.3922, 0.4302]
over_refuse = [0, 0.2959, 0.1735, 0.0775]
refuse = [0, 0.2933, 0.1664, 0.1149]
wrong = [0.4150, 0.2058, 0.2679, 0.3774]

# 设置柱的位置
x = np.arange(len(methods))

# 设置柱的宽度
width = 0.5

# 创建一个绘图
fig, ax = plt.subplots()

# 绘制堆叠柱状图
ax.bar(x, correct, width, label='Correct', color='green')
ax.bar(x, over_refuse, width, bottom=correct, label='Over Refuse', color='yellow')
ax.bar(x, refuse, width, bottom=np.array(correct) + np.array(over_refuse), label='Refuse', color='orange')
ax.bar(x, wrong, width, bottom=np.array(correct) + np.array(over_refuse) + np.array(refuse), label='Wrong', color='red')

# 添加标签、标题等
ax.set_xlabel('Methods')
ax.set_ylabel('Values')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()

plt.savefig('./picture/OOD_compare_overRefuse.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()
