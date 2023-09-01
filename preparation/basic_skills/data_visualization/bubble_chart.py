# 导库
import numpy as np
import matplotlib.pyplot as plt

# 气泡图（Bubble Chart）
'''
    （1）特点：类似散点图，但点的大小还代表附加数值
    （2）用途：同时展示两个维度的数据信息，强调第三维度的差异
'''

# 生成示例数据
x = np.random.rand(50)                          # 随机生成50个x坐标
y = np.random.rand(50)                          # 随机生成50个y坐标
size = np.random.randint(100, 500, 50)          # 随机生成50个气泡的大小
color = np.random.rand(50)                      # 随机生成50个气泡的颜色

# 创建Figure对象和子图
fig, ax = plt.subplots()

# 绘制气泡图，使用函数与散点图一致
'''
    cmap参数设置（摘自B站UP主hypothesis_hy的视频）：
    1. 'viridis'：从蓝色到黄色渐变，用于连续数据，特别适用于数据的渐变效果；
    2. 'plasma'：从紫色到橙色渐变，用于连续数据，较viridis颜色更丰富；
    3. 'inferno'：从黑色到黄色渐变，用于连续数据，较viridis颜色更加明亮；
    4. 'magma'：从黑色到白色渐变，用于连续数据，较viridis颜色更适合打印；
    5. 'cividis'：从蓝色到黄色渐变，用于连续数据，颜色较温和；
    6. 'cool'：从青色到蓝色渐变，用于连续数据，适用于冷色调的数据；
    7. 'hot'：从黑色到红色渐变，用于连续数据，适用于暖色调的数据；
    8. 'coolwarm'：从蓝色到红色渐变，用于连续数据，冷暖色调交替；
    9. 'rainbow'：七色彩虹渐变，用于连续数据，颜色多样
'''
bubble = ax.scatter(x, y, s=size, c=color, cmap='coolwarm', alpha=0.7, edgecolors='k')

# 添加颜色条
cbar = plt.colorbar(bubble)

# 设置图表标题
ax.set_title('Bubble Chart', fontsize=14)

# 设置刻度标签字体大小
ax.tick_params(axis='both', which='major', labelsize=10)

# 添加网格线
ax.grid(True, linestyle='--', alpha=0.7)

# 设置背景颜色
ax.set_facecolor('#f0f0f0')

# 设置图表边框颜色
ax.spines['top'].set_color('None')
ax.spines['right'].set_color('None')

# 显示图像
plt.tight_layout()
plt.show()