# numpy常用语法

# 导库
import numpy as np

# 排序
x = np.array([3, 1, 2])
np.argsort(x)

# 所有元素相乘
a = np.prod([[1.,2.],[3.,4.]])   # 1 * 2.0 * 3 * 4.0 = 24.0
print(a)