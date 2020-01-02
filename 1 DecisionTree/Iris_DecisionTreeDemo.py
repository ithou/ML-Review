from sklearn.datasets import load_iris
from sklearn import tree
import graphviz

# 加载数据
iris = load_iris()
# ID3 算法决策树
clf = tree.DecisionTreeClassifier(criterion='entropy')

feature_names = iris.feature_names
target_names = iris.target_names

# 训练
clf = clf.fit(iris.data, iris.target)

# 数据集
input = [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [4, 3, 2, 1]]

for index in range(len(input)):
    pre_data = [input[index]]  # 向量 而非数组
    predict_id = clf.predict(pre_data)[0]  # 注意这里是 clf.predict()[0]
    predict_name = target_names[predict_id]
    print("预测数据: " + str(pre_data))
    print("花名: " + str(predict_name) + '\n')

print("模型评价: " + str(clf.score(iris.data, iris.target)))


dot_data = tree.export_graphviz(clf, out_file="test",
                     feature_names=feature_names,
                     class_names=target_names,
                     filled=True, rounded=True)
graph = graphviz.Source(dot_data)