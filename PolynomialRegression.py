# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 14:02:08 2019

@author: USER
"""

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

#reading the data

df=pd.read_csv("FuelConsumptionCo2.csv")
df.head()

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Engine Size")
plt.ylabel("emission")
plt.show()

#training and testing dataset

msk = np.random.rand(len(df))<0.8
train =cdf[msk]
test = cdf[~msk]

#polynomial

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)
train_x_poly

#now use linearregression

clf = linear_model.LinearRegression()
train_y = clf.fit(train_x_poly, train_y)

#the coefficients

print('Coefficients: ',clf.coef_)
print('Intercept :' ,clf.intercept_)

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
XX= np.arange(0.0,10.0,0.1)
YY = clf.intercept_[0] + clf.coef_[0][1]*XX+ clf.coef_[0][2]*np.power(XX, 2)
plt.plot(XX,YY, '-r')
plt.xlabel('Engine Size')
plt.ylabel('Emissions')


#evaluation

from sklearn.metrics import r2_score

test_x_poly = poly.fit_transform(test_x)
test_y_ = clf.predict(test_x_poly)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )
























 