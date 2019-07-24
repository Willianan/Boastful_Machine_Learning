# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019-07-24 20:23
@Project:Boastful_Machine_Learning
@Filename:svmLinearInseparble.py
@description:
    线性不可分SVM
"""



import matplotlib.pyplot as plt
import numpy as np


# 固定随机种子数
np.random.seed(2)
x_n = np.r_[np.random.randn(20,2) - [1,1],np.random.randn(20,2) + [1,1]]
y_n = [0] * 20 + [1] * 20

fig,ax = plt.subplots(figsize=(8,6))
ax.scatter(x_n[0:20,1],x_n[0:20,0],s=30,c='b',marker='o',label='y = 0')
ax.scatter(x_n[20:40,1],x_n[20:40,0],s=30,c='r',marker='x',label='y = 1')
ax.legend()
plt.show()

# 线性不可分SVM算法的学习
from sklearn.svm import SVC
clf_n = SVC(kernel='linear')
clf_n.fit(x_n,y_n)
# 获取训练结果并预测
print(clf_n.coef_)
print(clf_n.support_vectors_)
print(clf_n.predict(x_n))
print(clf_n.score(x_n,y_n))

# 重新设置松弛变量，实现SVM算法的学习
clf_nSV = SVC(kernel='linear',C=0.2)
clf_nSV.fit(x_n,y_n)
# 获取训练结果并预测
print(clf_nSV.coef_)
print(clf_nSV.support_vectors_)
print(clf_nSV.predict(x_n))
print(clf_nSV.score(x_n,y_n))