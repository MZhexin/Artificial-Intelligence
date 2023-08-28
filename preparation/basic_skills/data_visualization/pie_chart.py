# 导库
import matplotlib.pyplot as plt

# 饼图（Pie Chart）
# 参考教程：https://www.bilibili.com/video/BV1wP411k7NM?p=6
'''
    （1）特点：扇形表示数据占比，用于显示分类数据的比例关系
    （2）用途：展示数据的百分比构成，用于显示相对比例和份额
'''

# 生成示例数据
labels1 = ['Group 1', 'Group 2', 'Group 3']
labels2 = ['Group 4', 'Group 5', 'Group 6']
sizes1 = [30, 20, 50]
sizes2 = [25, 35, 30]   # 这里存放数据，饼图的百分比是按这些数值在总量中的占比自动计算出来的

# 设置颜色列表
colors1 = ['blue', 'orange', 'green']
colors2 = ['red', 'purple', 'brown']

# 创建Figure对象和子图
fig, ax = plt.subplots()

# 绘制饼图
wedges1, texts1, autotexts1 = ax.pie(sizes1,                                # 输入的数据
                                     pctdistance=0.75,                      # 调整文字的位置（避免与内圈文字重叠）
                                     labels=labels1,                        # 标签列表
                                     colors=colors1,                        # 颜色列表
                                     autopct='%1.1f%%',                     # 得到小数的参考样式，此处设计为一位小数
                                     startangle=90,                         # 开始绘制的起始角度
                                     wedgeprops=dict(edgecolor='w'))        # 边框设置
wedges2, texts2, autotexts2 = ax.pie(sizes2,
                                     colors=colors2,
                                     radius=0.5,                            # 半径大小
                                     autopct='%1.1f%%',
                                     startangle=90,
                                     wedgeprops=dict(edgecolor='w'))

# 设置文本标签字体大小和颜色
for autotext in autotexts1 + autotexts2:
    autotext.set_fontsize(15)
    autotext.set_color('white')

# 添加图例
ax.legend(wedges1 + wedges2, labels1 + labels2, loc='best')

# 设置图表标题
ax.set_title('Scatter Plot', fontsize=14)

# 设置背景颜色
ax.set_facecolor('#f0f0f0')

# 显示图像
plt.tight_layout()
plt.show()
