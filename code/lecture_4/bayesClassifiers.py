# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019-07-23 15:44
@Project:Boastful_Machine_Learning
@Filename:bayesClassifiers.py
@description:
    贝叶斯分类算法
"""


from __future__ import division

import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB


import sys
sys.path.append("G:/PycharmProjects/Boastful_Machine_Learning/util")
import common_function as func


# 载入数据集
CSV_FILE_PATH = "G:/PycharmProjects/Boastful_Machine_Learning/data/6_credit.csv"
df = func.readData(CSV_FILE_PATH)

# 数据可视化
a1 = df[df['credit'] == 1]
a2 = df[df['credit'] == 2]
a3 = df[df['credit'] == 3]
fig,ax = plt.subplots(figsize=(8,5))
# 构建a1的散点图
ax.scatter(a1['income'],a1['points'],s=30,c='b',marker='o',label='credit=1')
# 构建a2的散点图
ax.scatter(a2['income'],a2['points'],s=30,c='r',marker='x',label='credit=2')
# 构建a3的散点图
ax.scatter(a3['income'],a3['points'],s=30,c='g',marker='^',label='credit=3')
ax.legend()
ax.set_xlabel("income")
ax.set_ylabel("points")
plt.show()

fig,ax = plt.subplots(figsize=(8,5))
# 构建a1的散点图
ax.scatter(a1['house'],a1['numbers'],s=30,c='b',marker='o',label='credit=1')
# 构建a2的散点图
ax.scatter(a2['house'],a2['numbers'],s=30,c='r',marker='x',label='credit=2')
# 构建a3的散点图
ax.scatter(a3['house'],a3['numbers'],s=30,c='g',marker='^',label='credit=3')
ax.legend(loc="upper left")
ax.set_xlabel("house")
ax.set_ylabel("numbers")
plt.show()

# 数据划分
x = df.iloc[:,1:6]
y = df.credit
x_train,x_test,y_train,y_test = func.dividingData(x,y)

# 高斯朴素贝叶斯估计
GNB = GaussianNB()
GNB.fit(x_train,y_train)
# 获取各个类标记对应的先验概率
print(GNB.class_prior_)
# 获取各类标记对应的训练样本数
print(GNB.class_count_)
# 获取各类标记在各个特征值上的均值
print(GNB.theta_)
# 获取各个类标记在各个特征上的方差
print(GNB.sigma_)

# 预测
y_pred = GNB.predict(x_test)
print(y_pred)

# 评价
from sklearn.metrics import confusion_matrix   # 计算混淆矩阵，主要来评估分类的准确性
from sklearn.metrics import accuracy_score     # 计算精度得分
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_true=y_test,y_pred=y_pred))

print("====================================================")


# 逻辑回归
from sklearn.linear_model.logistic import LogisticRegression
clf = LogisticRegression(solver='liblinear',multi_class='auto')
clf.fit(x_train,y_train)

y_pred_classifier = clf.predict(x_test)
print(accuracy_score(y_test,y_pred_classifier))
print(confusion_matrix(y_true=y_test,y_pred=y_pred_classifier))



"""
			多项式朴素贝叶斯分类算法
"""

from sklearn.naive_bayes import MultinomialNB

MNB = MultinomialNB(alpha=1.0)
MNB.fit(x_train,y_train)
print(MNB.class_log_prior_)
print(MNB.intercept_)

# 预测
y_pred_MNB = MNB.predict(x_test)
print(accuracy_score(y_test,y_pred_MNB))
print(confusion_matrix(y_true=y_test,y_pred=y_pred_MNB))


"""
   伯努利朴素贝叶斯分类算法
"""
from sklearn.naive_bayes import BernoulliNB

BNB = BernoulliNB(alpha=1.0,binarize=2.0,fit_prior=True)
BNB.fit(x_train,y_train)
print(BNB.class_log_prior_)
print(BNB.feature_log_prob_)

# 预测
y_pred_BNB = BNB.predict(x_test)

accuracy_score(y_test,y_pred_BNB)
confusion_matrix(y_true=y_test,y_pred=y_pred_BNB)
