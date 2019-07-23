# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019-07-23 14:29
@Project:Boastful_Machine_Learning
@Filename:logisticRegression.py
@description:
    逻辑回归——梯度下降法实现
"""



from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

import sys
sys.path.append("G:/PycharmProjects/Boastful_Machine_Learning/util")
import common_function as func


# 构建sigmoid函数
def sigmoid(z):
	return 1 / (1 + np.exp(-z))

# predict函数
def predict(theta,x):
	prob = sigmoid(x * theta.T)
	return [1 if a>=0.5 else 0 for a in prob]

# gradientDescent函数
def gradientDescent(x,y,theta,alpha,m,numIter):
	"""
	:param x: input variable
	:param y: input variable
	:param theta: parameter
	:param alpha: learning rate
	:param m: number of samples
	:param numIter: number of iteration
	:return: theta parameter
	"""
	# 矩阵转置
	x_trans = x.transpose()
	for i in range(numIter):
		# 将theta转化为矩阵
		theta = np.matrix(theta)
		# 将预测值转化为数组
		pred = np.array(predict(theta,x))
		# 预测值-实际值
		loss = pred - y
		# 计算梯度
		gradient = np.dot(x_trans,loss) / m
		# 参数更新
		theta = theta - alpha * gradient
	return theta


# 载入数据集
CSV_FILE_PATH = "G:/PycharmProjects/Boastful_Machine_Learning/data/5_logisitic_admit.csv"
df = func.readData(CSV_FILE_PATH)
df.insert(1,"Ones",1)
print(df.head(10))
positive = df[df['admit'] == 1]
negative = df[df['admit'] == 0]

# 数据可视化
fig,ax = plt.subplots(figsize=(8,5))
ax.scatter(positive['gre'],positive['gpa'],s=30,c='b',marker='o',label='admit')
ax.scatter(negative['gre'],negative['gpa'],s=30,c='r',marker='x',label='not admit')
ax.legend()
ax.set_xlabel('gre')
ax.set_ylabel('gpa')
plt.show()

# 数据集划分
x = df.iloc[:,1:4]
y = df['admit']
x = np.array(x.values)
y = np.array(y.values)
m,n = np.shape(x)
print(m,n)
theta = np.ones(n)
print(x.shape,theta.shape,y.shape)


# 参数初始化
numIter = 1000
alpha = 0.00001

# 梯度下降法求解参数
theta = gradientDescent(x,y,theta,alpha,m,numIter)
print(theta)

# 预测并计算准确率
# 采用predict函数来预测y
pred = predict(theta,x)
# 将预测为1实际也为1，预测为0实际也为0的均记为1
correct = [1 if ((a == 1 and b == 1) or (a==0 and b == 0)) else 0 for (a,b) in zip(pred,y)]
# 采用加总correct值来计算预测对的个数
accuracy = (sum(map(int,correct)) % len(correct))
print("accuracy = {:.2f}%".format(100 * accuracy / m))
print("=================================================================")




LR = LogisticRegression(solver='liblinear')
LR.fit(x,y)
print(LR.coef_)
pred_sk = LR.predict(x)
# 将预测为1实际也为1，预测为0实际也为0的均记为1
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(pred_sk, y)]
# 采用加总correct值来计算预测对的个数
accuracy = (sum(map(int,correct)) % len(correct))
print("accuracy = {:.2f}%".format(100 * accuracy / m))
print(pred_sk)
print(LR.score(x, y))