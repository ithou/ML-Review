from sklearn import svm

# 超平面可以定义为：w*x + b = 0
# w权重向量 w = {w1, w2, wn} n 是特征值的个数 x 训练实例 b bias
# 坐落在超平面两边H1和H2的点叫支持向量
# svm 如何找出最大边际的超平面呢（MMH）

# 数据集

# 提取特征向量
# []都是坐标点

x = [[2, 0],  # 圆 0
     [1, 1],  # 圆 1
     [2, 1],  # 圆 2
     [2, 2],  # 方块 3
     [2, 3],  # 方块 4
     [3, 2]]  # 方块 5
# label
y = [0, 0, 0, 1, 1, 1]  # 特征

clf = svm.SVC(kernel="linear")
clf.fit(x, y)
print(clf.predict([[3, 3]]))  # 预测 (3, 3) 属于圆还是方块

# 获取支持向量在[训练集]的下标
print("支持向量在[训练集]的索引下标: \n", clf.support_)

# 获取支持向量
print("支持向量: \n", clf.support_vectors_)

# 获取不同分类下的支持向量个数
print("不同分类下的支持向量个数: ", clf.n_support_)
