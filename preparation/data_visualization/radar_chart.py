# 导库
import matplotlib.pyplot as plt
import numpy as np

# 雷达图（Rader Chart）
'''
    （1）特点：多维数据的可视化图像，用多边形表示数据分布
    （2）用途：显示多个维度数据相对比较，可视化数据特征和优劣
'''

# 生成示例数据：每个维度的数据值
categories = ['A', 'B', 'C', 'D', 'E']      # 维度名称
values = [4, 2, 3, 5, 4]                   # 第数据

# 将第一个维度值赋值到列表的最后，以实现闭合
values += values[:1]

# 计算每个维度对应的角度（把整圆等分成N份，N是所要分成的维度个数）
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)

# 将角度与数值对应起来
angles = np.concatenate((angles, [angles[0]]))

# 创建Figure对象和子图，注意这里用的是极坐标
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

# 绘制雷达图并填充颜色
ax.plot(angles, values, linewidth=1.5, linestyle='--', color='b', label='Data')
ax.fill(angles, values, alpha=0.3, color='b')

# 添加图例
ax.legend(loc='best', fontsize=12)

# 设置标题
ax.set_title('Radar Chart', fontsize=16, fontweight='bold')

# 设置角度刻度（即维度的标签）
ax.tick_params(axis='both', labelsize=12)

# 设置背景颜色
ax.set_facecolor('#f0f0f0')

# 设置边框颜色
ax.spines['polar'].set_color('#b0b0b0')

# 显示图像
plt.tight_layout()
plt.show()