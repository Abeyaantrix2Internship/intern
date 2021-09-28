# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 07:32:54 2019

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# read data
dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#splitting data into training set and test set
#from sklearn.cros_validation import train_test_split
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#spliting is not required as we have very less no of samples

# fitting with Linear regression
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)

#fitting with polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=10) # change degree and see results begining value 2
x_poly=poly_reg.fit_transform(x)

# new linear regression object ##  polynomial linear regression model
poly_lin=LinearRegression()
poly_lin.fit(x_poly,y)

#plotting

plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title(' plot linear regression')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

# poly reg plot


plt.scatter(x,y,color='red')
plt.plot(x,poly_lin.predict(poly_reg.fit_transform(x)),color='blue')
plt.title(' plot polynomial regression')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()
# scaled plot
x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color='red')
plt.plot(x_grid,poly_lin.predict(poly_reg.fit_transform(x)),color='blue')
plt.title(' polynomial regression')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()



lin_reg.predict(6.5)

poly_lin.predict(poly_reg.fit_transform(6.5))