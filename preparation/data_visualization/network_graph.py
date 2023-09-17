# 导库
import networkx as nx
import matplotlib.pyplot as plt

# 网络图（Network Graph）
# 推荐绘图网站：https://csacademy.com/app/graph_editor/
'''
    （1）特点：节点和边构成的图形，用于显示关系网络
    （2）用途：可视化节点和边的关系网络，显示节点的中心性和连接性
'''

# 创建一个简单的网络图
G = nx.Graph()                                                                  # 实例化一个图
G.add_nodes_from([1, 2, 3, 4, 5, 6])                                            # 创建节点
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 1), (1, 3), (4, 6)])      # 创建边

# 绘制网络图
pos = nx.circular_layout(G)     # 设置节点的位置

# 调整节点和边的样式
nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=1000, label='Nodes')
nx.draw_networkx_edges(G, pos, edge_color='gray', width=2)
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

# 设置图例
plt.legend(loc='best')

# 设置标题
plt.title('Network Graph', fontsize=16, fontweight='bold')

# 设置背景颜色（plt.gca()函数是获得当前轴的意思，即获取当前的axes对象）
plt.gca().set_facecolor('#f0f0f0')

# 隐藏坐标轴
plt.axis('off')

# 显示图像
plt.show()