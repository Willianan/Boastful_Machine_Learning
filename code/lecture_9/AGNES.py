# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019-07-25 16:40
@Project:Boastful_Machine_Learning
@Filename:AGNES.py
@description:
    层次聚类：AGNES算法
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import AgglomerativeClustering


# 载入数据集
df = pd.read_csv("G:/PycharmProjects/Boastful_Machine_Learning/data/11_beverage.csv")
x = df.iloc[:,0:2]
x = np.array(x.values)

# AGNES算法
n_clusters = 4
# 设定算法为AGNES算法，距离度量为最小距离
ward = AgglomerativeClustering(n_clusters=n_clusters,linkage='ward')
ward.fit(x)

# 输出相关聚类结果，并评估聚类效果
labels = ward.labels_
print("各类簇标签值：",labels)
from sklearn import metrics
y_pred = ward.fit_predict(x)
print(metrics.calinski_harabasz_score(x,y_pred))


# 可视化
markers = ['o','^','*','s']
colors = ['r','b','g','peru']
plt.figure(figsize=(7,5))
# 画没干过类簇的样本点
for c in range(n_clusters):
	cluster = x[labels == c]
	plt.scatter(cluster[:,0],cluster[:,1],marker=markers[c],c=colors[c],s=20)
plt.xlabel('juice')
plt.ylabel('sweet')
plt.show()





# 设定算法AGNES算法，距离度量为最大距离
complete = AgglomerativeClustering(n_clusters=n_clusters,linkage='complete')
complete.fit(x)

# 输出相关聚类结果，并评估聚类效果
labels_com = complete.labels_
print("各类簇标签值：",labels_com)
from sklearn import metrics
y_pred_com = complete.fit_predict(x)
print(metrics.calinski_harabasz_score(x,y_pred_com))


# 可视化
markers = ['o','^','*','s']
colors = ['r','b','g','peru']
plt.figure(figsize=(7,5))
# 画没干过类簇的样本点
for c in range(n_clusters):
	cluster = x[labels == c]
	plt.scatter(cluster[:,0],cluster[:,1],marker=markers[c],c=colors[c],s=20)
plt.xlabel('juice')
plt.ylabel('sweet')
plt.show()

