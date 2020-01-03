import random
import numpy as np


# 创建数据
def getData(num, bias, variance):  # 实例（行数）、偏向、方差
    x = np.zeros(shape=(num, 2))  # num x 2的数组
    y = np.zeros(shape=num)  # 归类标签
    for i in range(0, num):
        x[i][0] = 0
        x[i][1] = i
        y[i] = x[i][0] + x[i][1] + bias
    return x, y


# 梯度下降算法
# 参数：实例 x， 分类标签 y，要学习的参数 theta，学习率 alpha，实例个数，迭代次数
def gradientDescent(x, y, theta, alpha, m, numIterations):
    xTrans = np.transpose(x)
    for i in range(0, numIterations):
        y_heat = np.dot(x, theta)  #
        loss = y_heat - y

        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration: " + str(i) + " | Cost: " + str(cost))

        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        # 更新法则
        theta = theta - alpha * gradient
    return theta


x, y = getData(5, 2, 1)
print("x: %s" % format(x))
print("y: %s" % format(y))

m, n = np.shape(x)
n_y = np.shape(y)

numIeration = 10000  # 迭代次数
alpha = 0.0005
theta = np.ones(n)
theta = gradientDescent(x, y, theta, alpha, m, numIeration)
print(theta)


