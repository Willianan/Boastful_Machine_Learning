# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019-07-24 15:25
@Project:Boastful_Machine_Learning
@Filename:randomForest.py
@description:
    随机森林
"""




import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


import sys
sys.path.append("G:/PycharmProjects/Boastful_Machine_Learning/util")
import common_function as func


# 导入数据集
CSV_FILE_PATH = "G:/PycharmProjects/Boastful_Machine_Learning/data/7_traffic.csv"
df = func.readData(CSV_FILE_PATH)


# 数据可视化
x = df.iloc[:,0:6]
y = df.traffic
x.hist(xlabelsize=12,ylabelsize=12,figsize=(18,12))
x.hist(xlabelsize=12,ylabelsize=12,figsize=(8,5))
plt.show()


# 数据划分
x_train,x_test,y_train,y_test = func.dividingData(x,y)


# ID3决策树算法进行训练
from sklearn import tree
tree_ID3 = tree.DecisionTreeClassifier(criterion='entropy')
tree_ID3.fit(x_train,y_train)
# 预测
y_pred_ID3 = tree_ID3.predict(x_test)


# 评估ID3算法
print(accuracy_score(y_test,y_pred_ID3))
print(confusion_matrix(y_test,y_pred_ID3))


# 随机森林训练
from sklearn.ensemble import RandomForestClassifier
# 定义一个随机森林分类器
clf = RandomForestClassifier(n_estimators=10,max_depth=None,
                             min_samples_split=2,oob_score=True,random_state=0)
clf.fit(x_train,y_train)
print(clf.oob_score_)

# 根据训练结果对测试集进行预测
y_pred_rf = clf.predict(x_test)
# 评估
print(accuracy_score(y_test,y_pred_rf))
print(confusion_matrix(y_test,y_pred_rf))

# 用ExtraTreeClassifier进行训练
from sklearn.ensemble import ExtraTreesClassifier
# 定义一个极端随机森林分类器
clf_extra = ExtraTreesClassifier(n_estimators=10,max_depth=None,min_samples_split=2,random_state=0)
clf_extra.fit(x_train,y_train)

#预测与评估
y_pred_extra = clf_extra.predict(x_test)
print(accuracy_score(y_test,y_pred_extra))
print(confusion_matrix(y_test,y_pred_extra))
