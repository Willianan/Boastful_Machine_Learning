# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019-07-26 15:29
@Project:Boastful_Machine_Learning
@Filename:housePriceForecast.py
@description:
    项目实战
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



from sklearn import metrics
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

'''
数据预处理
'''
# 数据加载与预览
CSV_FILE_PATH = "G:/PycharmProjects/Boastful_Machine_Learning/data"

df = pd.read_csv(CSV_FILE_PATH + "/13_house_train.csv")
print(df.head())
print(df.info())
print(df.describe())

# 缺失值处理
print(df[df.isnull().values == True])
df = df.fillna(df.mean())
print(df.loc[95])

# 数据转换
df['built_date'] = pd.to_datetime(df['built_date'])
print(df.head())

import datetime as dt
now_year = dt.datetime.today().year
age = now_year - df.built_date.dt.year
print(df.pop('built_date'))
print(df.insert(2,'age',age))
print(df.head())

print(df['floor'].unique())
df.loc[df['floor'] == 'Low','floor'] = 0
df.loc[df['floor'] == 'Medium','floor'] = 1
df.loc[df['floor'] == 'High','floor'] = 2
print(df.head())
print(df.info())

'''
特征提取
'''

# 变量特征图表
# 直方图
df.hist(xlabelsize=8,ylabelsize=8,layout=(3,5),figsize=(20,12))
# 变量箱线图
df.plot(kind='box',subplots=True,layout=(3,5),sharex=False,sharey=False,fontsize=12,figsize=(20,12))
plt.show()

# 变量关联性分析
corr_matrix = df.corr()
print(corr_matrix['price'].sort_values(ascending=False))
# 目标变量与特征变量的散点图
plt.figure(figsize=(8,3))
plt.subplot(121)
plt.scatter(df['price'],df['area'])
plt.subplot(122)
plt.scatter(df['price'],df['pm25'])
plt.figure(figsize=(8,3))
plt.subplot(121)
plt.scatter(df['price'],df['age'])
plt.subplot(122)
plt.scatter(df['price'],df['green_rate'])
plt.show()


'''
建模训练
'''
def showRMSE(name,y_ture,y_pred):
	print("RMSE_" + name + "：",np.sqrt(metrics.mean_squared_error(y_ture,y_pred)))


# 对数据集的划分
# 选取特征变量并划分数据集
col_n = ['area','crime_rate','pm25','traffic','shockproof','school','age','floor']
x = df[col_n]
y = df.price

from sklearn.model_selection import train_test_split
x = np.array(x.values)
y = np.array(y.values)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=1)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

# 采用不同算法的建模训练
# 岭回归算法
ridge = linear_model.Ridge(alpha=0.1)
ridge.fit(x_train,y_train)

y_pred_ridge = ridge.predict(x_test)
showRMSE('Ridge',y_test,y_pred_ridge)


# lasso回归算法
lasso = linear_model.Lasso(alpha=0.1)
lasso.fit(x_train,y_train)

y_pred_lasso = lasso.predict(x_test)
showRMSE("Lasso",y_test,y_pred_lasso)

# 支持向量机回归
linear_svr = SVR(kernel='linear')
linear_svr.fit(x_train,y_train)

y_pred_svr = linear_svr.predict(x_test)
showRMSE('SVR',y_test,y_pred_svr)

# 随机森林回归
rf = RandomForestRegressor(random_state=200,max_features=0.3,n_estimators=10)
rf.fit(x_train,y_train)

y_pred_rf = rf.predict(x_test)
showRMSE('rf',y_test,y_pred_rf)

# 参数调优
# SVM参数调优
alphas_svr = np.linspace(0.1,1.2,20)
rmse_svr = []
for alpha in alphas_svr:
	model = SVR(kernel='linear',C=alpha)
	model.fit(x_train,y_train)
	y_hat = model.predict(x_test)
	rmse_svr.append(np.sqrt(metrics.mean_squared_error(y_test,y_hat)))
plt.plot(alphas_svr,rmse_svr)
plt.title("Cross Validation Score with Model SVR")
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.show()

# Lasso回归的参数调优
alphas_lasso = np.linspace(-0.1,0.1,20)
rmse_lasso = []
for alpha in alphas_lasso:
	model = linear_model.Lasso(alpha=alpha)
	model.fit(x_train,y_train)
	y_hat = model.predict(x_test)
	rmse_lasso.append(np.sqrt(metrics.mean_squared_error(y_test,y_hat)))
plt.plot(alphas_lasso,rmse_lasso)
plt.title("Cross Validation Score with Model Lasso")
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.show()




df_test = pd.read_csv(CSV_FILE_PATH + "/13_house_test.csv")
print(df_test.head())
print(df_test.info())

df_test['built_date'] = pd.to_datetime(df_test['built_date'])
age = now_year - df_test.built_date.dt.year

df_test.pop('built_date')
df_test.insert(2,'age',age)

df_test.loc[df_test['floor'] == 'Low','floor'] = 0
df_test.loc[df_test['floor'] == 'Medium','floor'] = 1
df_test.loc[df_test['floor'] == 'High','floor'] = 2
print(df_test.head())


testX = df_test[col_n]
svr_test = SVR(kernel='linear',C=1.0)
svr_test.fit(x,y)
testy_pred = svr_test.predict(testX)

submit = pd.read_csv(CSV_FILE_PATH+"/13_pred_test.csv")
submit['price'] = testy_pred
print(submit.head())