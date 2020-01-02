# https://www.jianshu.com/p/4cabef90f98b

from sklearn.feature_extraction import DictVectorizer
import csv
import os
from sklearn import preprocessing
from sklearn import tree


def getData():
    with open(r'H:\Project\ML-Review\1 DecisionTree\data.csv', 'r', encoding='UTF-8') as f:
        data = f.readlines()
    f.close()
    return data


# 读取数据
csvData = getData()
reader = csv.reader(csvData)
headers = next(reader)  # 第一次调用reader 就指向第一行 再调用next() 就指向第二行数据

# 标题
print("标题 %s \n" % format(headers))

# 特征值List
featureList = []
# 类别List , Yes/No
labelList = []

# 从表格的第二行开始，一行一行的循环整个cvs
for row in reader:
    labelList.append(row[len(row)-1])  # 表格最后一列class_buys_computer的内容
    rowDict = {}
    for i in range(1, len(row)-1):   # 不读取第一列RID
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)

print(labelList)
print(rowDict)

# 将特征值列表中的内容转化成向量形式
# 转换features为独热(one-hot)
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList) .toarray()

print(dummyX)
print("dummyX:\n %s \n" % format(dummyX))
print("features_name: %s \n" % format(vec.get_feature_names()))

# 将标签类的内容转化成向量形式 使用python自带LabelBinarizer
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummyY:\n %s \n" % format(dummyY))

# 使用tree分类器创建,使用信息熵 ID3 算法
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)
print("clf: \n %s" % format(clf))

# 利用原来的数据生成新的数据进行预测
# 取第一行
oneRowX = dummyX[0, :]
print("oneRowX: " + str(oneRowX))

# 修改数据作为测试
newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
print("newRowX: " + str(newRowX))

# 预测
predictedY = clf.predict([newRowX])  # 传入的是一个向量[]
print(predictedY)
