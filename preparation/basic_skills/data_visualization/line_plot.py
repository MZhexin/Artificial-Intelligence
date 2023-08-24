# 导库
import matplotlib.pyplot as plt

# 折线图（Line Plot）
# 参考教程：https://www.bilibili.com/video/BV1wP411k7NM?p=2
'''
    （1）特点：连续数据的变化趋势，直线连接数据点
    （2）用途：显示数据随时间或其他连续变量的趋势，检测趋势和周期性
'''

# 创建数据：5条折线，每条折线有5个数据点
y1 = [2, 4, 1, 5, 2]
y2 = [1, 3, 2, 3, 4]
y3 = [3, 1, 3, 2, 5]
y4 = [4, 2, 3, 1, 3]
y5 = [2, 3, 4, 2, 1]
x = [i for i in range(len(y1))]

# 创建一个Figure对象和一个子图
fig, ax = plt.subplots()

# 设置折线的线条样式、颜色和标签（因为有5条折线，因此需要5个元素的列表）
'''
    1. 线条样式有'-', '--', ':', '-.', ' ',  '', 'None', 'solid', 'dashed', 'dashdot', 'dotted'
    2. 线条颜色也可以用'#123456'表示
    3. 线条标签即图例名称
'''
line_styles = ['-', '--', ':', '-.', 'dashdot']                         # 线条样式
line_colors = ['b', 'g', 'r', 'y', 'black']                             # 线条颜色
line_labels = ['Line 1', 'Line 2', 'Line 3', 'Line 4', 'Line 5', ]      # 线条标签
line_widths = [1.5, 2.0, 2.5, 1.0, 1.5]                                 # 线条粗细

# 利用循环，绘制y1、y2、y3、y4和y5的折线
# globals()是全局函数，返回一个字典，y1至y5是字典的键
# globals()[f'y{i+1}']返回对应折线中数据点的值
for i in range(5):
    ax.plot(x, globals()[f'y{i+1}'],
            linestyle=line_styles[i],
            color=line_colors[i],
            label=line_labels[i],
            linewidth=line_widths[i])

# 用annotate方法添加数据点的具体数值（只添加y1的值）
for i in range(len(x)):
    ax.annotate(f'{y1[i]}',                     # 注释文本的内容
                (x[i], y1[i]),                  # 被注释数据点的坐标
                textcoords='offset points',     # 注释文本的坐标系属性，'offset points'表示偏移量的单位是点
                xytext=(0, 10),                 # 注释文本的坐标
                ha='center')                    # 注释文本的水平对齐方式

# 添加图例
ax.legend()

# 添加横纵坐标的标签
ax.set_xlabel('X Axis', fontsize=12)
ax.set_ylabel('Y Axis', fontsize=12)

# 设置图表标题
ax.set_title('Line Plot')

# 设置刻度标签大小
ax.tick_params(axis='both', which='major', labelsize=10)

# 添加网格线
ax.grid(True, linestyle='--', alpha=0.7)

# 设置背景颜色
ax.set_facecolor('#f0f0f0')

# 设置图表边框颜色
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

# 显示图像
plt.show()