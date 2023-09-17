# 置信区间
# 在曲线上下界再各画一条曲线，并给区间填充颜色即可

# 导库
import numpy as np
import matplotlib.pyplot as plt

# 示例数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 上下界
confidence_interval = 0.2
upper_bound = y + confidence_interval
lower_bound = y - confidence_interval

# 创建Figure对象和子图
plt.figure(figsize=(10, 6))
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Confidence Intervals')

# 绘制线条
plt.plot(x, y, label='Data', color='blue')

# 置信区间填充颜色
plt.fill_between(x, lower_bound, upper_bound,       # 填充范围：在lower_bound和upper_bound之间填充颜色
                 color='lightblue',                 # 颜色
                 alpha=0.5,                         # 透明度
                 label='Confidence Interval')       # 图例标签

# 添加图例
plt.legend()

# 显示图像
plt.show()