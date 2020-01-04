import numpy as np


# k-means算法
# x:数据集(每行代表一个数据点，每列代表一个特征值) k:分类数 maxIt:迭代次数
def kmeans(X, k, maxIt):
    numPonits, numDim = X.shape  # 返回数据集的行、列数
    dataSet = np.zeros((numPonits, numDim + 1))  # 初始化新的数据集dataSet 比X多一列，用来存放分标签
    dataSet[:, :-1] = X  # 所有行的第一列 - 倒数第二列都为X

    # 随机选取中心点 所有数据点 随机选k行
    centroids = dataSet[np.random.randint(numPonits, size=k), :]
    # 中心点的最后一列初始化值(类标签)：1到k
    centroids[:, -1] = range(1, k + 1)

    iterations = 0  # 循环次数
    oldCentroids = None  # 旧的中心点

    # 没有停止迭代的话 (False)
    while not shouldStop(oldCentroids, centroids, iterations, maxIt):
        print("当前循环迭代次数: ", iterations)
        print("当前数据集: \n", dataSet)
        print("当前中心点: \n", centroids)

        # 为什么用copy而不是 = 因为后面会做修改 oldCentrods和centrods是两部分内容
        # 不让彼此的修改而影响彼此
        oldCentroids = np.copy(centroids)
        iterations += 1

        # 更新类标签
        updateLabels(dataSet, centroids)
        # 更新中心点
        centroids = getCentroids(dataSet, k)

    return dataSet

def shouldStop(oldCentroids, centroids, iterations, maxIt):
    if iterations > maxIt:
        return True
    return np.array_equal(oldCentroids, centroids)  # 比较新旧中心点的值是否相等，相等返回True，迭代终止

# 更新类标签
def updateLabels(dataSet, centroids):
    numPoints, numDim = dataSet.shape  #获取当前数据集的行列数
    for i in range(0, numPoints):
        dataSet[i, -1] = getLabelFromClosesCentroid(dataSet[i, :-1], centroids)


def getLabelFromClosesCentroid(dataSetRow, centroids):
    label = centroids[0, -1]  # 初始化本条数据类标签为第一个中心点的类标签
    minDis = np.linalg.norm(dataSetRow - centroids[0, :-1])  # 调用内嵌的方法算距离 一直在更新
    for i in range(1, centroids.shape[0]):  # 求与每个中心点之间的距离
        dis = np.linalg.norm(dataSetRow - centroids[i, :-1])
        if dis < minDis:
            minDis = dis
            label = centroids[i, -1]
    print("最小距离minDist: ", minDis)
    return label


# 根据归类后数据集和k值，计算新的中心点
def getCentroids(dataSet, k):
    # 最后返回的新的中心点的值有k行，列数与dataSet相同
    result = np.zeros((k, dataSet.shape[1]))
    for i in range(1, k + 1):
        # 取出标记为i的数据(除最后一列)
        oneCluster = dataSet[dataSet[:, -1] == i, :-1]
        result[i - 1, :-1] = np.mean(oneCluster, axis=0)
        result[i-1, -1] = i
    return result


x1 = np.array([1, 1])
x2 = np.array([2, 1])
x3 = np.array([4, 3])
x4 = np.array([5, 4])
testX = np.vstack((x1, x2, x3, x4))

result = kmeans(testX, 2, 10)
print("result: \n", result)