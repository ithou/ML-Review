import numpy as np
import pylab as pl
from sklearn import svm

# 1.准备数据集
np.random.seed(42)
class_A_X = np.random.randn(5, 2) - [2, 2]
class_B_X = np.random.randn(5, 2) + [2, 2]
# r c
# 2.提取特征变量
X = np.r_[class_A_X, class_B_X]  # 将每一行矩阵拼接成一块
Y = [0] * 5 + [1] * 5  # label 贴标签
print("Y: \n", Y)

# 3.结合算法
clf = svm.SVC(kernel='linear')  # 分类器clf 核函数为linear
clf.fit(X, Y)  # 训练

# 获取权重向量 w
w = clf.coef_[0]
print("回归系数: \n", w)
print("支持向量: \n", clf.support_vectors_)
w0 = w[0]
w1 = w[1]
w2 = clf.intercept_[0]

# 斜率
k = -(w0 / w1)
# 截距
b = -(w2 / w1)

# 准备绘图数据
xx = np.linspace(-5, 5)
# 超平面上的y
yy = k * xx + b

# 超平面计算
support_point_down = clf.support_vectors_[0]
yy_down = k * xx + (support_point_down[1] - k * support_point_down[0])
support_point_up = clf.support_vectors_[-1]
yy_up = k * xx + (support_point_up[1] - k * support_point_up[0])

# 绘图
pl.plot(xx, yy, 'k-')
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')

pl.scatter(X[:, 0],
           X[:, 1],
           c=Y)
pl.show()
