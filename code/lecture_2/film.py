# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019-07-22 22:17
@Project:Boastful_Machine_Learning
@Filename:film.py
@description:
    影厅观影人数的拟合
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score


CSV_FILE_PATH = 'G:/PycharmProjects/Boastful_Machine_Learning/data/3_film.csv'
df = pd.read_csv(CSV_FILE_PATH)
print(df.head())

df.hist(xlabelsize=12,ylabelsize=12,figsize=(12,7))
plt.show()

# 绘制密度图
df.plot(kind='density',subplots=True,layout=(2,2),sharex=False,fontsize=8,figsize=(12,7))
plt.show()

# 绘制箱线图
df.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False,fontsize=8,figsize=(12,7))
plt.show()

# 多变量的数据的相关系数热力图
names = ['filmnum','filmsize','ratio','quality']
correlations = df.corr()
# 绘制相关系数热力图
fig = plt.figure()
ax = fig.add_subplot((111))
cax = ax.matshow(correlations,vmin=0.3,vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,4,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()

# 散点图矩阵
scatter_matrix(df,figsize=(8,8),c='b')
plt.show()

# 选取特征变量与响应变量，并进行数据划分
x = df.iloc[:,1:4]
y = df.filmnum
# 把x，y转化为数组形式
x = np.array(x.values)
y = np.array(y.values)
# 以25%的数据构建测试样本，剩余作为训练样本
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=1)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


# 进行线性回归
lr = LinearRegression()
lr.fit(x_train,y_train)
print("求解截距项：",lr.intercept_)
print("求解系数为：",lr.coef_)

# 根据求出的参数对测试集进行预测
y_hat = lr.predict(x_test)
print(y_hat[0:9])

plt.figure(figsize=(10,6))
t = np.arange(len(x_test))
# 绘制y_test曲线
plt.plot(t,x_test,'r',linewidth=2,label='y_test')
# 绘制y_hat曲线
plt.plot(t,y_hat,'b',linewidth=2,label='y_train')
plt.legend()
plt.show()

# 对预测进行评价
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