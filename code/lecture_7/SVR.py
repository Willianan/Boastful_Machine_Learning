# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019-07-24 21:03
@Project:Boastful_Machine_Learning
@Filename:SVR.py
@description:
    支持向量回归机SVR
"""


import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("G:/PycharmProjects/Boastful_Machine_Learning/util")
import common_function as func

np.random.seed(4)
x_1 = np.random.randn(40,2)
y_1 = x_1[:,0] + 2 * x_1[:,1] + np.random.randn(40)

fig,ax = plt.subplots(figsize=(8,6))
ax.scatter(y_1,x_1[:,0],s=30,c='b',marker='o')
ax.scatter(y_1,x_1[:,1],s=30,c='b',marker='x')
plt.show()


# SVM算法是学习
from sklearn.svm import SVR
clf_1 = SVR(gamma='auto')
clf_1.fit(x_1,y_1)

# 获取训练结果并预测
print(clf_1.support_vectors_)
print(clf_1.score(x_1,y_1))

# 预测值与实际值可视化
y_hat = clf_1.predict(x_1)
t = np.arange(len(x_1))
func.curveShow(t,y_1,y_hat)

# 评估
from sklearn import metrics
# 用scikit-learn计算MAE
print("MAE: ",metrics.mean_absolute_error(y_1,y_hat))
# 用scikit-learn计算MsE
print("MSE: ",metrics.mean_squared_error(y_1,y_hat))
# 用scikit-learn计算RMSE
print("RMSE: ",np.sqrt(metrics.mean_squared_error(y_1,y_hat)))

