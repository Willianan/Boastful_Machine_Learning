# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019-07-25 21:06
@Project:Boastful_Machine_Learning
@Filename:PCA.py
@description:
    PCA降维
"""





import numpy as np

x = [0.69,-1.31,0.39,0.09,1.29,0.49,0.19,-0.81,-0.31,-0.71]
y = [0.49,-1.21,0.99,0.29,1.09,0.79,-0.31,-0.81,-0.31,-1.01]

x = np.c_[x,y]
print(x)

# PCA训练
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(x)
# 输出相关PCA训练结果
print("特征值：",pca.explained_variance_)
print("特征值的贡献率：",pca.explained_variance_ratio_)

# 保留主成分为1进行PCA的训练
pca_one = PCA(n_components=1)
pca_one.fit(x)
print("特征值：",pca_one.explained_variance_)
print("特征值的贡献率：",pca_one.explained_variance_ratio_)
# 生成降维后的数据
x_new = pca_one.transform(x)
print(x_new)
