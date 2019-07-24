# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019-07-24 19:43
@Project:Boastful_Machine_Learning
@Filename:svmLinearSeparable.py
@description:
    支持向量机
"""

# 线性可分SVM

import matplotlib.pyplot as plt
import numpy as np


np.random.seed(0)
# 按列连接两个矩阵
x = np.r_[np.random.randn(20,2) - [2,2],np.random.randn(20,2) + [2,2]]
# 生成类别变量y
y = [0] * 20 + [1] * 20

fig,ax = plt.subplots(figsize=(8,6))
# 构建y=0的散点图
ax.scatter(x[0:20,1],x[0:20,0],s=30,c='b',marker='o',label="y = 0")
# 构建y=1的散点图
ax.scatter(x[20:40,1],x[20:40,0],s=30,c='r',marker='x',label="y = 1")
ax.legend()
plt.show()

# SVM算法的学习
from sklearn.svm import SVC
clf = SVC(kernel='linear')
clf.fit(x,y)

# 获取训练结果并预测
print(clf.coef_)
print(clf.support_vectors_)
print(clf.predict(x))
print(clf.score(x,y))

# 绘制超平面与支持向量
# 获取参数w
w = clf.coef_[0]
# 获取斜率
a = -w[0] / w[1]
xx = np.linspace(-5,5)
yy = a * xx - (clf.intercept_[0]) / w[1]

b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

plt.plot(xx,yy,'k-')
plt.plot(xx,yy_down,'k--')
plt.plot(xx,yy_up,'k--')

plt.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],
            c='black',s=30,facecolors='none')
# 构建y=0的散点图
plt.scatter(x[0:20,1],x[0:20,0],s=30,c='b',marker='o',label="y = 0",cmap=plt.cm.Paired)
# 构建y=1的散点图
plt.scatter(x[20:40,1],x[20:40,0],s=30,c='r',marker='x',label="y = 1",cmap=plt.cm.Paired)
plt.legend()
plt.show()

