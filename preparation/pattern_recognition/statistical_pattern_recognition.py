# 作业1：统计模式识别
# 根据已知男女学生的身高体重，预测未知性别体检数据的性别

# 导库
import numpy as np
import matplotlib.pyplot as plt

# 数据
data_list = np.array([[170, 68], [130, 66], [180, 71], [190, 73], [160, 70],
                    [150, 66], [190, 68], [210, 76], [100, 58], [170, 75],
                    [140, 62], [150, 64], [120, 66], [150, 66], [130, 65],
                    [140, 70], [150, 60], [145, 65], [160, 75]])
label_list = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 2, 2, 2, 2])

# 可视化
fig, ax = plt.subplots()
colors = ['tab:blue', 'tab:orange', 'tab:green']
markers = ['o', 's', 'D']
labels = ['female', 'male', 'unknown']
for i in range(3):
        data = data_list[label_list == i]
        ax.scatter(data[:, 0],
                    data[:, 1],
                    marker=markers[i],
                    color=colors[i],
                    label=labels[i])
ax.legend(loc='best')
ax.set_title('Students\' Information', fontsize=14)
ax.set_xlabel('Height')
ax.set_ylabel('Weight')
ax.tick_params(axis='both', which='major', labelsize=10)
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_facecolor('#f0f0f0')
plt.tight_layout()
plt.show()

# 根据身高和体重，利用线性回归方法区分性别
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
known_data = data_list[: -4, :]
known_label = label_list[: -4]
w = np.dot(np.dot(np.linalg.inv(np.dot(known_data.T, known_data)), known_data.T), known_label)  # 最小二乘法
def function(x, y):
    result = w[0] * x + w[1] * y
    return result
X, Y = np.meshgrid(known_data[:,0], known_data[:,1])
Z = function(X, Y)
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='black', alpha=0.8)
ax.scatter(known_data[:,0], known_data[:,1], known_label)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')
ax.grid(True)
ax.set_xlabel('Height')
ax.set_ylabel('Weight')
ax.set_zlabel('Gender')
ax.set_xlim(100, 180)
ax.set_ylim(55, 80)
ax.set_zlim(0, 1)
ax.view_init(elev=45, azim=-20)
plt.show()

unknown_data = data_list[-4:]
for i in range(len(unknown_data)):
    result = function(unknown_data[i][0], unknown_data[i][1])
    if result <= 0.5:
        print('第{0}位同学的性别预测为{1}'.format(i + 1, '女'))
    else:
        print('第{0}位同学的性别预测为{1}'.format(i + 1, '男'))