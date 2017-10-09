# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 09:59:42 2017

@author: zzd
"""
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


#数据可视化
def plotFigure(X,y,X_test,yp):
    plt.figure()
    plt.scatter(X,y,c='k',label='data')
    plt.plot(X_test,yp,c='r',label='max_depth=5',linewidth=2)
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Decision Tree Regression')
    plt.legend()
    plt.show()

x=np.linspace(-5,5,200)
siny=np.sin(x)
X=np.mat(x).T
y=siny+np.random.rand(1,len(siny))*1.5
y=y.tolist()[0]


clf=DecisionTreeRegressor(max_depth=4)#max_depth选取最大的树深度，类似先剪枝
clf.fit(X,y)

X_test=np.arange(-5,5,0.05)[:,np.newaxis]
yp=clf.predict(X_test)

plotFigure(X.tolist(),y,X_test,yp)


