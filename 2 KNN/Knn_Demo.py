from sklearn import neighbors
from sklearn.datasets import load_iris

# 导入iris数据
iris = load_iris()
# KNN分类器
knn = neighbors.KNeighborsClassifier(n_neighbors=5)  # 临近的节点数量，默认值是5

# 训练模型
knn.fit(iris.data, iris.target)

# 特征数组
feature_names = iris.feature_names
# 分类数组
target_names = iris.target_names

input = [[0.1, 0.2, 0.3, 0.4],
         [1, 2, 3, 4],
         [4, 5, 6, 7]]

for row in range(len(input)):
    predict_id = knn.predict([input[row]])[0]
    predict_name = target_names[predict_id]
    print(predict_name)
# output: setosa versicolor virginica

# 模型评估
score = knn.score(iris.data, iris.target)
print(score)  # 0.9666666666666667

# 算正确率
X_data = iris.data
Y_data = iris.target

num_true = 0
for idx in range(len(X_data)):
    if Y_data[idx] == knn.predict([X_data[idx]])[0]:  # 预测的数据为向量 [[x x x x]]
        num_true += 1
    idx += 1

print("KNN 正确率：" + str((num_true / float(len(X_data))) * 100) + '%')




















