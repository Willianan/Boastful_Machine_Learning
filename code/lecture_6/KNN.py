# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019-07-24 16:27
@Project:Boastful_Machine_Learning
@Filename:KNN.py
@description:
    k近算法
"""



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


import sys
sys.path.append("G:/PycharmProjects/Boastful_Machine_Learning/util")
import common_function as func


# 导入数据集
CSV_FILE_PATH = "G:/PycharmProjects/Boastful_Machine_Learning/data/7_traffic.csv"
df = func.readData(CSV_FILE_PATH)

# 数据集划分
x = df.iloc[:,0:6]
y = df.traffic
# 把x,y转化为数组形式
x = np.array(x.values)
y = np.array(y.values)
# 以25%的数据构建测试样本，其余为训练样本
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# k邻近算法的训练样本
from sklearn import neighbors
# 定义KNN算法分类器
KNN = neighbors.KNeighborsClassifier(n_neighbors=5,weights='distance')
KNN.fit(x_train,y_train)

# 预测与评估
y_pred_knn = KNN.predict(x_test)
print(y_pred_knn)
print(accuracy_score(y_test,y_pred_knn))
print(confusion_matrix(y_test,y_pred_knn))
print("------------------------------------------")

# 设置k=15
# 定义KNN算法分类器
KNN1 = neighbors.KNeighborsClassifier(n_neighbors=15,weights='distance')
KNN1.fit(x_train,y_train)

# 预测与评估
y_pred_knn1 = KNN1.predict(x_test)
print(y_pred_knn1)
print(accuracy_score(y_test,y_pred_knn1))
print(confusion_matrix(y_test,y_pred_knn1))
print("=============================================")


# 余弦相似性
# 定义KNN算法分类器
KNN2 = neighbors.KNeighborsClassifier(n_neighbors=15,metric='cosine',weights='distance')
KNN2.fit(x_train,y_train)

# 预测与评估
y_pred_knn2 = KNN.predict(x_test)
print(y_pred_knn2)
print(accuracy_score(y_test,y_pred_knn2))
print(confusion_matrix(y_test,y_pred_knn2))