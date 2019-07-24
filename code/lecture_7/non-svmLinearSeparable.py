# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019-07-24 20:54
@Project:Boastful_Machine_Learning
@Filename:non-svmLinearSeparable.py
@description:
    非线性可分SVM
"""




import matplotlib.pyplot as plt
import numpy as np


np.random.seed(3)
# 按列连接两个矩阵
x_Sq = np.r_[np.random.randn(20,2)**2 - [1,1],np.random.randn(20,2)**2 + [1,1]]
# 生成类别变量y
y_Sq = [0] * 20 + [1] * 20

fig,ax = plt.subplots(figsize=(8,6))
# 构建y=0的散点图
ax.scatter(x_Sq[0:20,1],x_Sq[0:20,0],s=30,c='b',marker='o',label="y = 0")
# 构建y=1的散点图
ax.scatter(x_Sq[20:40,1],x_Sq[20:40,0],s=30,c='r',marker='x',label="y = 1")
ax.legend()
plt.show()

# SVM算法的学习
from sklearn.svm import SVC
clf = SVC(kernel='poly',degree=2,gamma='auto')
clf.fit(x_Sq,y_Sq)

# 获取训练结果并预测
print(clf.support_vectors_)
print(clf.predict(x_Sq))
print(clf.score(x_Sq,y_Sq))

# SVM算法的学习,设置核函数为径向基
from sklearn.svm import SVC
clf = SVC(kernel='rbf',gamma=1)
clf.fit(x_Sq,y_Sq)

# 获取训练结果并预测
print(clf.support_vectors_)
print(clf.predict(x_Sq))
print(clf.score(x_Sq,y_Sq))


# SVM算法的学习,设置核函数为sigmoid函数
from sklearn.svm import SVC
clf = SVC(kernel='sigmoid',gamma=1)
clf.fit(x_Sq,y_Sq)

# 获取训练结果并预测
print(clf.support_vectors_)
print(clf.predict(x_Sq))
print(clf.score(x_Sq,y_Sq))