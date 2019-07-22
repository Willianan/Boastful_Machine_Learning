# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019-07-22 17:23
@Project:Boastful_Machine_Learning
@Filename:Boston.py
@description:
    波士顿房屋价格的拟合与预测
"""




import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn import metrics
from sklearn.metrics import r2_score
import math

# 中文支持
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

boston = load_boston()
print(boston.keys())
print(boston.feature_names)

bos = pd.DataFrame(boston.data)
print(bos[5].head())

bos_target = pd.DataFrame(boston.target)
print(bos_target.head())

x = bos.iloc[:,5:6]
y = bos_target
plt.scatter(x,y)
plt.xlabel('住宅平均房间数')
plt.ylabel('房屋价格')
plt.title('RM与MEDV的关系')
plt.show()

x = np.array(x.values)
y = np.array(y.values)

# 以25%的数据构建测试样本，剩余作为训练样本
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

lr = LinearRegression()
lr.fit(x_train,y_train)
print("求解系数为：",lr.intercept_)
print("求解系数为：",lr.coef_)

y_hat = lr.predict(x_test)
print(y_hat[0:9])

# y_test与y_hat的可视化
plt.figure(figsize=(10,6))
t = np.arange(len(x_test))
# 绘制曲线
plt.plot(t,y_test,'r',linewidth=2,label='y_test')
plt.plot(t,y_hat,'b',linewidth=2,label='y_train')
plt.legend()
plt.show()

# 拟合优度R2的输出方法一
print("r2: ",lr.score(x_test,y_test))
# 拟合优度R2的输出方法二
print("r2: ",r2_score(y_test,y_hat))
# 用scikit-learn计算MAE
print("MAE: ",metrics.mean_absolute_error(y_test,y_hat))
# 用scikit-learn计算MsE
print("MSE: ",metrics.mean_squared_error(y_test,y_hat))
# 用scikit-learn计算RMSE
print("RMSE: ",np.sqrt(metrics.mean_squared_error(y_test,y_hat)))
print()

# 最小二乘法求解
def linefit(x,y):
	N = len(x)
	sx,sy,sxx,syy,sxy=0,0,0,0,0
	for i in range(N):
		sx += x[i]
		sy += y[i]
		sxx += x[i] * x[i]
		syy += y[i] * y[i]
		sxy += x[i] * y[i]
	a = (sy * sx / N - sxy) / (sx * sx / N - sxx)
	b = (sy - a * sx) / N
	return a,b

a,b = linefit(x_train,y_train)
y_hat1 = a * x_test + b

# 用scikit-learn计算MAE
print("MAE: ",metrics.mean_absolute_error(y_test,y_hat1))
# 用scikit-learn计算MsE
print("MSE: ",metrics.mean_squared_error(y_test,y_hat1))
# 用scikit-learn计算RMSE
print("RMSE: ",np.sqrt(metrics.mean_squared_error(y_test,y_hat1)))