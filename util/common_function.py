# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019-07-23 11:05
@Project:Boastful_Machine_Learning
@Filename:common_function.py
@description:
    函数
"""

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score


# 读取数据
def readData(CSV_FILE_PATH):
	df = pd.read_csv(CSV_FILE_PATH)
	print(df.head())
	return df


# 数据划分
def dividingData(x,y):
	# 把x,y转化为数组形式
	x = np.array(x.values)
	y = np.array(y.values)
	# 以25%的数据构建测试样本，其余为训练样本
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
	print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
	return x_train, x_test, y_train, y_test


# 作图函数
def curveShow(t,y_test,y_hat):
	"""
	:param t: t变量
	:param y_test: 测试集真实值
	:param y_hat: 测试集预测值
	"""
	plt.figure(figsize=(10,6))
	# 绘制y_test曲线
	plt.plot(t, y_test, 'r', linewidth=2, label='y_test')
	# 绘制y_hat曲线
	plt.plot(t, y_hat, 'b', linewidth=2, label="y_train")
	plt.legend()
	plt.show()

# 对预测结果进行评价
def evaluation(y_test,y_hat):
	# 拟合优度R2的输出方法
	print("r2_score: ",r2_score(y_test,y_hat))
	# 用scikit-learn计算MAE
	print("MAE: ",metrics.mean_absolute_error(y_test,y_hat))
	# 用scikit-learn计算MsE
	print("MSE: ",metrics.mean_squared_error(y_test,y_hat))
	# 用scikit-learn计算RMSE
	print("RMSE: ",np.sqrt(metrics.mean_squared_error(y_test,y_hat)))