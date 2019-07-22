# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019-07-22 16:39
@Project:Boastful_Machine_Learning
@Filename:random_number.py
@description:
    Random number
"""




import matplotlib.pyplot as plt
import numpy as np



x1 = np.random.randint(-10,10,100)

y1 = x1**2 - x1 + np.random.randint(1,100,100)

plt.scatter(x1,y1,c='b')
plt.show()