# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019-07-25 15:37
@Project:Boastful_Machine_Learning
@Filename:k-means.py
@description:
    K均值算法
"""




import matplotlib.pyplot as plt
import numpy as np



from sklearn import metrics

import sys
sys.path.append("G:/PycharmProjects/Boastful_Machine_Learning/util")
import common_function as func


# 载入数据集
CSV_FILE_PATH = "G:\PycharmProjects\Boastful_Machine_Learning/data/11_beverage.csv"
df = func.readData(CSV_FILE_PATH)


# 样本数据转化并可视化
x = df.iloc[:,0:2]
x = np.array(x.values)

plt.scatter(x[:,0],x[:,1],s=20,marker='o',c='b')
plt.xlabel("juice")
plt.ylabel('sweet')
plt.show()


# k均值算法的训练，并输出结果
from sklearn.cluster import KMeans
n_clusters = 3
kmean = KMeans(n_clusters=n_clusters)
kmean.fit(x)


# 输出相关聚类结果，并评估聚类效果
y_pred = kmean.predict(x)
print(metrics.calinski_harabasz_score(x,y_pred))

labels = kmean.labels_
centers = kmean.cluster_centers_
print("各类簇标签值：",labels)
print("各类簇中心：",centers)


# 聚类结果及其各类簇中心点的可视化
marker = ['o','^','*']
colors = ['r','b','g']
plt.figure(figsize=(7,5))
# 画每个类簇的样本点
for c in range(n_clusters):
	cluster = x[labels == c]
	# 按照c的不同取值选取相应样本点、标记、颜色，画散点图
	plt.scatter(cluster[:,0],cluster[:,1],marker=marker[c],s=20,c=colors[c])
# 画出每个类簇中心点
plt.scatter(centers[:,0],centers[:,1],marker='o',c='black',alpha=0.9,s=50)
plt.xlabel("juice")
plt.ylabel('sweet')
plt.show()
print("=====================================")


# 设置k=4进行k均值算法的训练，并输出结果
n_clusters_four = 4
kmean_four = KMeans(n_clusters=n_clusters_four)
kmean_four.fit(x)
y_pred_four = kmean_four.predict(x)
print(metrics.calinski_harabasz_score(x,y_pred_four))
# 输出聚类中心点
labels_four = kmean_four.labels_
centers_four = kmean_four.cluster_centers_
print("各类簇中心：",centers_four)
# 聚类结果及其各类簇中心点的可视化
markers = ['o','^','*','s']
colors = ['r','b','g','peru']
plt.figure(figsize=(7,5))
# 画每个类簇的样本点
for c in range(n_clusters_four):
	cluster = x[labels_four == c]
	# 按照c的不同取值选取相应样本点、标记、颜色，画散点图
	plt.scatter(cluster[:,0],cluster[:,1],marker=markers[c],s=20,c=colors[c])
# 画出每个类簇中心点
plt.scatter(centers_four[:,0],centers_four[:,1],marker='o',c='black',alpha=0.9,s=50)
plt.xlabel("juice")
plt.ylabel('sweet')
plt.show()
print("------------------------------------------")

# 设置k一定的取值范围，进行聚类并评价不同的聚类的结果
from scipy.spatial.distance import cdist
# 类簇的数量2~9
clusters = range(2,10)
# 距离函数
distances_sum = []
for k in clusters:
	kmean_model = KMeans(n_clusters=k).fit(x)
	# 计算各对象各类簇中心的欧式距离，生成距离表
	distances_point = cdist(x,kmean_model.cluster_centers_,'euclidean')
	# 提取每个对象到其类簇中心的距离，并相加
	distances_cluster = sum(np.min(distances_point,axis=1))
	# 依次存入类簇数从2到9的距离结果
	distances_sum.append(distances_cluster)
plt.plot(clusters,distances_sum,'bx-')
plt.xlabel("k")
plt.ylabel("distances")
plt.show()