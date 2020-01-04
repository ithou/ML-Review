import numpy as np
import matplotlib.pyplot as plt


def getLabelFromCentrals(row, centrals):
    label = centrals[0, -1]
    minDist = np.linalg.norm(row - centrals[0, :-1])
    for i in range(1, centrals.shape[0]):
        dist = np.linalg.norm(row - centrals[i, :-1])
        if dist < minDist:
            minDist = dist
            label = centrals[i, -1]
    return label


# 更新聚类中心
def updateCentrals(dataSets, k):
    centrals = np.zeros((k, dataSets.shape[1]))
    for i in range(1, k + 1):
        oneCluster = dataSets[dataSets[:, -1] == i, :-1]
        # axis=0,对列求平均值
        # axis=1,对行求平均值
        centrals[i - 1, :-1] = np.mean(oneCluster, axis=0)
        centrals[i - 1, -1] = i
    return centrals


# 判断是否需要继续循环
def canRun(oldCentrals, centrals, it, maxIt):
    # 判断聚类中心是否没改变
    if np.array_equal(oldCentrals, centrals):
        return False
    # 判断是否达到最大迭代次数
    return it < maxIt


def Kmeans(x, k, maxIt):
    num, numFutures = np.shape(x)
    dataSets = np.zeros((num, numFutures + 1))
    dataSets[:, :2] = x
    # 随机2个点的位置为聚类中心点
    centrals = dataSets[np.random.randint(num, size=k), :]
    centrals[:, -1] = range(1, k + 1)
    print("dataSets:\n", dataSets)
    print("centrals:\n", centrals)
    oldCentrals = None
    it = 1
    while canRun(oldCentrals, centrals, it, maxIt):
        print("=========", it, "=========")
        # 遍历每一个样本
        for i in range(0, num):
            # print(dataSets[i, :-1])
            # 样本与每一个central求距离，并更新标签(y|label)
            dataSets[i, -1] = getLabelFromCentrals(dataSets[i, :-1], centrals)
        oldCentrals = np.copy(centrals)
        centrals = updateCentrals(dataSets, k)
        print("dataSets:\n", dataSets)
        print("centrals:\n", centrals)
        it += 1
    return dataSets[:, -1]


# 聚类数量
k = 2
# 最大迭代次数
maxIt = 1000
x = np.array([[1, 1],
              [2, 1],
              [4, 3],
              [5, 4]])
print("data:\n", x)
pred_y = Kmeans(x, k, maxIt)
plt.scatter(x[:, 0], x[:, 1], c=pred_y)
plt.show()
