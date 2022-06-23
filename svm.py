from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import datetime
import time
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1.读取训练数据集
path = 'svm_train.txt'
datatrain = np.loadtxt(path, dtype=float, delimiter=' ')   # dtype是读出数据类型，delimiter是分隔符

# 2.读取测试数据集
path1 = 'svm_test.txt'
datatest = np.loadtxt(path1, dtype=float, delimiter=' ')

# 3.划分数据与标签
x, y = np.split(datatrain, indices_or_sections=(3,), axis=1)
# x为数据，y为标签，indices_or_sections表示从4列分割
x1, y1 = np.split(datatest, indices_or_sections=(3,), axis=1)
# x为数据，y为标签，axis=1表示列分割，等于0行分割

# # 1.读取训练数据集
# path = 'original_data_svm_train.txt'
# datatrain = np.loadtxt(path, dtype=float, delimiter=' ')   # dtype是读出数据类型，delimiter是分隔符
#
# # 2.读取测试数据集
# path1 = 'original_data_svm_test.txt'
# datatest = np.loadtxt(path1, dtype=float, delimiter=' ')
#
# # 3.划分数据与标签
# x, y = np.split(datatrain, indices_or_sections=(6,), axis=1)
# # x为数据，y为标签，indices_or_sections表示从4列分割
# x1, y1 = np.split(datatest, indices_or_sections=(6,), axis=1)
# # x为数据，y为标签，axis=1表示列分割，等于0行分割


# 3.训练svm分类器
classifier = svm.SVC(C=0.1, kernel='rbf', gamma='auto', decision_function_shape='ovr', probability=True)#0.05
# classifier = MLPClassifier()
# classifier = RandomForestClassifier(n_jobs=4, criterion='gini', n_estimators=200, min_samples_split=4, oob_score=True)
# classifier = KNeighborsClassifier(n_neighbors=5, n_jobs=4)
# classifier = ExtraTreesClassifier(n_jobs=4,  n_estimators=100, criterion='gini', min_samples_split=10,
#                        max_features=50, max_depth=40, min_samples_leaf=4)
# classifier = GaussianNB()

# ovr:一对多策略，ovo表示一对一
svm_train_start = time.time()
classifier.fit(x, y.ravel())     # ravel函数在降维时默认是行序优先
svm_train_end = time.time()
svm_train_time_str = "test run time: %.4f seconds" % (svm_train_end - svm_train_start)

svm_test_start = time.time()
y_ = classifier.predict(x1)
svm_test_end = time.time()
svm_test_time_str = "test run time: %.4f seconds" % (svm_test_end - svm_test_start)

print(svm_train_time_str)
print(svm_test_time_str)
print("训练集：", classifier.score(x, y))
print("测试集：", classifier.score(x1, y1))
a = pd.crosstab(index = y_,
            columns = y1[:,-1],
            rownames=['predict'],
            colnames=['True'],
            margins=True #统计
           )
print(a)

# yp = classifier.predict_proba(x1)
#
# with open('E:\\fire_detection\\tcn\pth\\tcn_app_svm_result.txt', 'w') as f:
#     for a, b in zip(y1.squeeze(), yp):
#         f.write(str(a) + ' ' + str(b[0]) + ' ' + str(b[1]) + ' ' + str(b[2]) + "\r")
# 4.计算svc分类器的准确率
# print("训练集：", classifier.score(x, y))
# print("测试集：", classifier.score(x1, y1))

'''
# 也可直接调用accuracy_score方法计算准确率
from sklearn.metrics import accuracy_score
tra_label = classifier.predict(train_data)  # 训练集的预测标签
tes_label = classifier.predict(test_data)  # 测试集的预测标签
print("训练集：", accuracy_score(train_label, tra_label))
print("测试集：", accuracy_score(test_label, tes_label))
'''
# 查看决策函数
# print('train_decision_function:\n', classifier.decision_function(x))  # 查看决策函数
# # print('train_data:\n', x)                                             # 查看训练数据
# # print('predict_data:\n', x1)                                          # 查看测试数据
# print('predict_result:\n', classifier.predict(x1))                    # 查看测试结果
# print('predict_data:\n', y1)


# 使用TSNE进行降维处理。从4维降至2维。
# tsne = TSNE(n_components=3, learning_rate=100).fit_transform(x1)
#
# # 设置画布的大小
# plt.figure(figsize=(12, 6))
#
# plt.scatter(tsne[:, 0], tsne[:, 1], c=y1)
#
# plt.colorbar()
# plt.show()