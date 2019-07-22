# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019-07-22 16:44
@Project:Boastful_Machine_Learning
@Filename:people_number.py
@description:
    影厅观影人数与影厅面积的关系
"""




import pandas as pd
import matplotlib.pyplot as plt


CSV_FILE_PATH = 'G:/PycharmProjects/Boastful_Machine_Learning/data/3_film.csv'
df = pd.read_csv(CSV_FILE_PATH)
x = df['filmsize']
y = df['filmnum']
plt.scatter(x,y,c='b')
plt.xlabel("x")
plt.ylabel("y")
plt.show()