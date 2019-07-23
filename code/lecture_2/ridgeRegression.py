# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019-07-23 10:43
@Project:Boastful_Machine_Learning
@Filename:ridgeRegression.py
@description:
    岭回归
"""

from __future__ import division
import numpy as np
from sklearn.linear_model import Ridge
import sys
sys.path.append("G:/PycharmProjects/Boastful_Machine_Learning/util")
import common_function as func


# 导入数据与划分数据集
CSV_FILE_PATH = 'G:/PycharmProjects/Boastful_Machine_Learning/data/3_film.csv'
df = func.readData(CSV_FILE_PATH)

# 数据划分
x = df.iloc[:,1:4]
y = df['filmnum']
x_train,x_test,y_train,y_test = func.dividingData(x,y)

# 岭回归估计
ridge = Ridge(alpha=0.1)
ridge.fit(x_train,y_train)
print("求解截距项：",ridge.intercept_)
print("求解系数为：",ridge.coef_)

# 对测试集进行预测
y_hat = ridge.predict(x_test)
print(y_hat[0:9])

t = np.arange(len(x_test))
func.curveShow(t,y_test,y_hat)
func.evaluation(y_test,y_hat)



