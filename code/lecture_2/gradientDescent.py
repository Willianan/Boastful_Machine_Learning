# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019-07-23 9:43
@Project:Boastful_Machine_Learning
@Filename:gradientDescent.py
@description:
    梯度下降法的实现
"""



from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import metrics




CSV_FILE_PATH = 'G:/PycharmProjects/Boastful_Machine_Learning/data/3_film.csv'
df = pd.read_csv(CSV_FILE_PATH)
print(df.head())

df.insert(1,'Ones',1)
print(df.head())

cols = df.shape[1]
x = df.iloc[:,1:cols]
y = df.filmnum

# 把x,y转化为数组形式
x = np.array(x.values)
y = np.array(y.values)
# 以25%的数据构建测试样本，其余为训练样本
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=1)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

# 构建计算成本函数
def computeCost(x,y,theta):
	inner = np.power(((x * theta.T) - y),2)
	return np.sum(inner) / (2 * len(x))

# 构建梯度下降法函数
def gradientDescent(x,y,theta,alpha,iters):
	"""
	:param x: input variable
	:param y: input variable
	:param theta: parameter
	:param alpha: learning rate
	:param iters:  number of iterations
	:return:theta,cost
	"""
	# 构建零值矩阵
	temp = np.matrix(np.zeros(theta.shape))
	# 计算需要求解的参数个数
	parameters = int(theta.ravel().shape[1])
	# 构建iters个0的数组
	cost = np.zeros(iters)
	for i in range(iters):
		# 计算theta.T * x - y
		error = (x * theta.T) - y
		# 对于theta中的每一个元素依次计算
		for j in range(parameters):
			# 计算两矩阵相乘
			term = np.multiply(error,x[:,j])
			# 更新规则
			temp[0,j] = theta[0,j] - ((alpha / len(x)) * np.sum(term))
		theta = temp
		cost[i] = computeCost(x,y,theta)
	return theta,cost

# 参数初始化
alpha = 0.000001
iters = 100
theta = np.matrix(np.array([0,0,0,0]))

# 采用gradientDescent()函数来优化求解
g,cost = gradientDescent(x,y,theta,alpha,iters)
print(g)

# 预测
y_hat = x_test * g.T
print(y_hat)

plt.figure(figsize=(10,6))
t = np.arange(len(x_test))
# 绘制y_test曲线
plt.plot(t,y_test,'r',linewidth=2,label='y_test')
# 绘制y_hat曲线
plt.plot(t,y_hat,'b',linewidth=2,label="y_train")
plt.legend()
plt.show()


# 对预测结果进行评价
# 拟合优度R2的输出方法
print("r2_score: ",r2_score(y_test,y_hat))
# 用scikit-learn计算MAE
print("MAE: ",metrics.mean_absolute_error(y_test,y_hat))
# 用scikit-learn计算MsE
print("MSE: ",metrics.mean_squared_error(y_test,y_hat))
# 用scikit-learn计算RMSE
print("RMSE: ",np.sqrt(metrics.mean_squared_error(y_test,y_hat)))