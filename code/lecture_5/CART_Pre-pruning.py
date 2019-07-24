# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019-07-24 14:57
@Project:Boastful_Machine_Learning
@Filename:CART_Pre-pruning.py
@description:
    CART算法预剪枝
"""



import numpy as np
import pydotplus


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from IPython.display import Image
from sklearn.externals.six import StringIO



import sys
sys.path.append("G:/PycharmProjects/Boastful_Machine_Learning/util")
import common_function as func


# 载入数据集
CSV_FILE_PATH = "G:/PycharmProjects/Boastful_Machine_Learning/data/7_buy.csv"
df = func.readData(CSV_FILE_PATH)

# 数据集预处理
x = df.iloc[:,0:4]
y = df.buy
x = np.array(x.values)
y = np.array(y.values)


# ID3算法分类
# 默认采用的是gini(cart算法)，通过entropy设置ID3算法
tree_CART_prun = DecisionTreeClassifier(max_depth=2)
tree_CART_prun.fit(x,y)
print(tree_CART)


# 预测与评估
y_pred_prun = tree_CART_prun.predict(x)
print(y_pred_prun)

print(accuracy_score(y,y_pred_prun))
print(confusion_matrix(y_true=y,y_pred=y_pred_prun))


# 生成决策树结构图
feature_names = list(df.columns[:-1])
target_names = ['0','1']
dot_data = StringIO()
tree.export_graphviz(tree_CART_prun,out_file=dot_data,feature_names=feature_names,class_names=target_names,
                     filled=True,rounded=True,special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())