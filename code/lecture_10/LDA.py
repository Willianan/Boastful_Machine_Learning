# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019-07-26 15:07
@Project:Boastful_Machine_Learning
@Filename:LDA.py
@description:
    线性判别分析(LDA)
"""



import numpy as np

x = [0.69,-1.31,0.39,0.09,1.29,0.49,0.19,-0.81,-0.31,-0.71]
y = [0.49,-1.21,0.99,0.29,1.09,0.79,-0.31,-0.81,-0.31,-1.01]

z = [0,0,0,0,0,1,1,1,1,1]

x = np.c_[x,y]
print(x)

# LDA训练
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# 设置LDA降维参数，并将降维后的维度设为1
lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(x,z)

# 输出相关LDA训练结果
x_new = lda.transform(x)
print("降维后变量：",x_new)
print("权重向量：",lda.coef_)


# 输出其他结果
print("每个类别的均值向量：",lda.means_)
print("整体样本的均值向量：",lda.xbar_)
cx