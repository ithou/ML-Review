# 处理分类型变量Demo

from sklearn import linear_model
from numpy import genfromtxt

path = r'./dataset2.csv'
data = genfromtxt(path, delimiter=',')
print("data:\n %s" % format(data))

# 取出 x_data
x_data = data[:, :-1]  # 除开最后一列的所有数据
print("x_data:\n %s" % format(x_data))
# 取出 y_data
y_data = data[:, -1]
print("y_data:\n %s" % format(y_data))

# 分类器
regr = linear_model.LinearRegression()
# 训练
regr.fit(x_data, y_data)

# 截距
intercept_ = regr.intercept_
print("截距为：%s" % format(intercept_))  # -0.3185446009389743

# 系数
coef_ = regr.coef_
print("系数为：%s" % format(coef_))  # [ 0.05262911  0.93497653 -0.26103286  0.40962441 -0.14859155]

# 多元线性回归方程为：y = -0.3185446009389743 + 0.05262911*英里数 + .. + -0.14859155*车型2

# 预测
# 果一个运输任务是跑102英里 运输6次，使用1类型的车，预计多少小时？
pre_data = [[102, 6, 0, 1, 0]]
y_pred = regr.predict(pre_data)
print(y_pred)  # [11.06910798]
