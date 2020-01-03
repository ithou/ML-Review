from numpy import genfromtxt
from sklearn import linear_model

path = r'./dataset1.csv'
data = genfromtxt(path, delimiter=',')
print(data)

x_data = data[:, :-1]  # 所有行 除开最后一列 最后一列为运输时间
print("x_data:\n %s" % format(x_data))

y_data = data[:, -1]  # 所有行 只取最后一列
print("y_data:\n %s " % format(y_data))

# 导入线性回归分类器
regr = linear_model.LinearRegression()
# 训练
regr.fit(x_data, y_data)

# 截距 b0
b0 = regr.intercept_
print(b0)  # -0.868701466781709

# 系数
b1 = regr.coef_
print(b1)  # [0.0611346  0.92342537]

# 公式: y = -0.86870 + 0.0611346*英里 + 0.92342537*次数

# 如果一个运输任务是跑102英里 运输6次，预计多少小时
x_pred = [[102, 6]]  # 向量
y_pred = regr.predict(x_pred)
print(y_pred)
