# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019-07-23 13:54
@Project:Boastful_Machine_Learning
@Filename:lasso.py
@description:
    Lasso回归
"""


from __future__ import division
from sklearn.linear_model import Lasso
import numpy as np
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

# lasso回归估计
lasso = Lasso(alpha=0.1)
lasso.fit(x_train,y_train)
print("求解截距项为：",lasso.intercept_)
print("求解系数为：",lasso.coef_)

# 预测
y_hat_lasso = lasso.predict(x_test)
print(y_hat_lasso[0:9])

# 图形可视化
t = np.arange(len(x_test))
func.curveShow(t,y_test,y_hat_lasso)

# 评价
func.evaluation(y_test,y_hat_lasso)

