# 导库
import matplotlib.pyplot as plt

# 柱状图（Bar Chart）
# 参考教程：https://www.bilibili.com/video/BV1wP411k7NM?p=3
'''
    （1）特点：长方形条形表示数据大小，用于比较不同类别的数据
    （2）用途：显示离散或分类数据的大小比较，展示排名和分组数据
'''

# 数据示例，假设有5个数据的高度
heights = [5, 7, 3, 4, 6]
x = [i for i in range(len(heights))]

# 创建Figure对象和子图
fig, ax = plt.subplots()

# 不同柱子的填充样式和颜色
patterns = ['/', 'x', '+', '-', 'o']                        # 填充图案（可以不添加）
colors = ['b', 'g', 'r', 'purple', 'orange']                # 填充颜色

# 绘制柱状图
width = 0.5     # 柱状图宽度

for i in range(len(heights)):
    bar = ax.bar(x[i], heights[i], width, color=colors[i], hatch=patterns[i], label='Bar {0}'.format(i + 1))
    height = bar[0].get_height()                                            # 得到每个柱子的高度
    ax.annotate('{0}'.format(height),                                       # 注释文本
                xy=(bar[0].get_x() + bar[0].get_width() / 2, height),       # 被注释点的坐标
                xytext=(0, 3),                                              # 注释坐标
                textcoords='offset points',                                 # 注释的坐标系属性
                ha='center',                                                # 注释的水平对齐方式
                va='bottom')                                                # 注释的垂直对齐方式

# 添加图例
ax.legend(fontsize=12, loc='upper right')

# 添加横纵坐标label
ax.set_xlabel('X Axis', fontsize=14, fontweight='bold')
ax.set_ylabel('Y Axis', fontsize=14, fontweight='bold')

# 设置整个图表的xy范围，避免过大或过小，而导致遮挡住重要信息
plt.ylim([0, 12])

#设置图表标题
ax.set_title('Bar Chart', fontsize=16, fontweight='bold')

# 设置刻度标签
ax.set_xticks(x)
ax.set_xticklabels(['Bar {}'.format(i + 1) for i in range(len(heights))], fontsize=12, fontweight='bold')

# 设置网格线
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 设置背景颜色
ax.set_facecolor('#f0f0f0')

# 去除图表边框
for spine in ax.spines.values():
    spine.set_visible(False)

# 显示图像
plt.show()
