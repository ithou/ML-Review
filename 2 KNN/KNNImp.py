# encoding:utf-8
import csv
import math
import operator

# 不借助sklearn.neighbors库实现KNN算法

# KNN计算步骤
# 1、载入所有样本的参数
# 2、未知样本与所有已知样本进行距离计算
# 3、根据距离从小到大升序
# 4、根据排序结果，选择前K个样本
# 5、根据选择的K个样本，遵循少数服从多数的原则，得到分类结果（投票原则）


# 加载数据
def loadDataSet(path, dataSet=[]):
    with open(path, 'r', encoding='UTF-8') as f:
        csvFiles = csv.reader(f)
        dataset_temp = list(csvFiles)  # 文件内容转换成list结构
    f.close()

    for index in range(len(dataset_temp) - 1):
        # 取除开最后一列的数据
        for x in range(4):
            # 重置数据（去掉最后一列）
            dataset_temp[index][x] = float(dataset_temp[index][x])
        dataSet.append(dataset_temp[index])  # 这里注意是dataset_temp[index]


# 计算实例之间的欧氏距离
def ComputeEucliDistance(input, param):
    total = 0  # 设置距离初始值为0
    # 计算所有维度的差的平方和
    for index in range(4):
        total += math.pow(input[index] - param[index], 2)  # 欧式平方距离
    return math.sqrt(total)  # 返回欧氏距离


# 测试集中的一个实例到训练集的距离最近的k个实例
# k：距离最近的个数
def getDistances(input, dataSet, k):
    distances = []
    for index in range(len(dataSet)):
        dist = ComputeEucliDistance(input, dataSet[index])
        distances.append((dataSet[index], dist))
    distances.sort(key=operator.itemgetter(1))
    result = []
    for index in range(k):
        result.append(distances[index][0][-1])
    return result


# 投票
def getResult(distances):
    """
     得到
     :param distances:附近的实例
     :return:得票最多的类别情况
    """
    classTimes = {}  # classVotes = {}
    for index in range(len(distances)):
        if distances[index] in classTimes:
            classTimes[distances[index]] += 1
        else:
            classTimes[distances[index]] = 1
        sortedClassTimes = sorted(classTimes.items(),
                                  key=operator.itemgetter(1),
                                  reverse=True)
        return sortedClassTimes[0][0]


if __name__ == '__main__':
    path = 'iris.csv'
    dataSet = []
    # 1
    loadDataSet(path, dataSet)
    # 234
    k = 5
    input = [[0.1, 0.2, 0.3, 0.4],
             [1, 2, 3, 4],
             [4, 5, 6, 7]]
    for index in range(len(input)):
        distances = getDistances(input[index], dataSet, k)
        # # 5
        result = getResult(distances)
        print(result)