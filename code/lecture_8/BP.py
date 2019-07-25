# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019-07-25 14:35
@Project:Boastful_Machine_Learning
@Filename:BP.py
@description:
    BP算法  手写数字的识别
"""




import numpy as np


from scipy.io import loadmat
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix



data = loadmat("G:/PycharmProjects/Boastful_Machine_Learning/data/10_digital.mat")
print(data)

x = data['X']
y = data['y']
print(x.shape,y.shape)
print(x[0,100:120])


# 数据预处理
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# 数据集的划分
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=2)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


# 设置MLP算法
mlp = MLPClassifier(solver='adam',activation='tanh',alpha=1e-5,
                    hidden_layer_sizes=(50,),learning_rate_init=0.001,max_iter=2000)
mlp.fit(x_train,y_train)


# 训练结果，并对测试集进行预测
print("每层网络层系数矩阵维度：\n",[coef.shape for coef in mlp.coefs_])
y_pred = mlp.predict(x_test)
print("预测结果：",y_pred)
print(mlp.intercepts_)


# 预测评估
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))