import numpy as np

def getData():
    with open('input.csv', 'r', encoding='UTF-8') as f:
        lines = f.readlines()
    f.close()
    x = []
    y = []
    for line in lines:
        x.append(int(line.split(",")[0]))
        y.append(int(line.split(",")[1].strip()))
    return x, y


def fitSLR(x, y):
    numerator = 0  # 分子
    denominator = 0  # 分母
    for index in range(0, len(x)):
        numerator += (x[index] - np.mean(x)) * (y[index] - np.mean(y))
        # denominator += (x[index] - np.mean(x)) ** 2
        denominator += np.square((x[index] - np.mean(x)))
    b1 = numerator / float(denominator)  # 系数 b1 = 分子 / 分母
    b0 = np.mean(y) - b1 * np.mean(x)  # 截距 b0 = y_bar - b1 * x_bar
    return b0, b1


# 预测函数
# y = b0 + b1 * x
def predict(x):
    y = b0 + x * b1
    return y


if __name__ == '__main__':
    x, y = getData()
    # x = [1,3,2,1,3]
    # y = [14,24,18,17,27]
    b0, b1 = fitSLR(x, y)
    y_hat = predict(6)

    print(y_hat)
