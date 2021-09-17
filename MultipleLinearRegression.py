# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 19:25:36 2021

@author: HP
"""
#%% importing lib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
print('library import completed')

#%% step 2 reading data
#Data=pd.read_csv('Salary_Data.csv')
Data=pd.read_csv('https://raw.githubusercontent.com/Abeyaantrix2Internship/intern/main/50_Startups.csv')

#%%
# Step 5 input and output
In1=Data.iloc[:,:-1].values # input variable
#In=Data.iloc[:,:4].values
Dp1=Data.iloc[:,4:5].values # Target Variable
#
#%% handling catogarical data one hot encodingg
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
H=OneHotEncoder()
CT = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
# 3 indicates column no 3 (state which has to one hot encoded)
# passthrough indicates only column mentioned is onehotencoded and remaining columns are kept as it is
In1= np.array(CT.fit_transform(In1), dtype = np.str)
#%% removing dummy column
In1=In1[:,1:]
#%% data type conversion  to make sure all columns has same data type
In1=In1.astype(np.float64)
Dp1=Dp1.astype(np.float64)

#%% split the data into train and test set
from sklearn.model_selection import train_test_split
In_train, In_test, Dp_train, Dp_test=train_test_split(In1, Dp1, test_size=0.25) # 25 for test 75 training


#%%
# Step 7
from sklearn.linear_model import LinearRegression
#step 8 creating the object, in this case no options set
LR=LinearRegression()
# step 9 train on training set
LR.fit(In_train,Dp_train) # will fit data onto the straight line, computes LR of DPN wrt to IND

#%%
NEWPred=LR.predict(In_test)
# our goal is to optimize regression model by eliminating insigninficant input vairables
# optimization methods: Backward elimination method
# 1 first we use all the input variables and compute the OLS
# 2 set the Significance level p( it tells how much the variables has influence(signinificant) 
#it has in predicting output) p=0.01 99%   0.1 -->90%
# std used value is .05  at 5% 95% confidance level
# 0.04, 0.06 
# 3 check the ols results and compare computed p value with the set p value accordingly discard 
# insignificant cariables (variables haveing p greater than the set p value)
# 4 compute the OLS with remaining variables and repeat steps 2 to 4

#%%
import statsmodels.formula.api as sm
import statsmodels.regression.linear_model as SM

# y=b0x0+b1x1+b2x2+b3x3+b4x4+b5x5

#y=b0x0+b1x1

I=np.append(arr=np.ones([50,1]).astype(int),values=In1,axis=1)
print(I)
#%%  iteration 1
Iopt=I[:,[0,1,2,3,4,5]]
#0->x0-11111         0<0.05
#1->x1-state1        0.953>0.05   highly singinifcant
#2->x2-state2        0.99>0.05     highly singinficant
#3->x3-RnD           0<0.05
#4->x4-Admin         0.6>0.05
#5->x5-Marketing     0.123>0.05

Optimizer=SM.OLS(endog=Dp1,exog=Iopt).fit()
Optimizer.summary()

#%% iteration 2

#0->x0-11111         0<0.05
#1->x1-state1        0.953>0.05   highly singinifcant  # removed
#2->x2-state2        0.99>0.05     highly singinficant# removed
#3->x3-RnD           0<0.05
#4->x4-Admin         0.6>0.05
#5->x5-Marketing     0.123>0.05
Iopt=I[:,[0,3,4,5]]
# y=b0x0+b1x1+b2x2+b3x3
#0->x0-11111         0<0.05           0<0.05   
#3->x3-RnD           0<0.05     x1    0<0.05
#4->x4-Admin         0.6>0.05   x2    0.602>0.5   admin column highly singin   
#5->x5-Marketing     0.123>0.05 x3    0.105>0.05
Optimizer=SM.OLS(endog=Dp1,exog=Iopt).fit()
Optimizer.summary()
#%%
#Iopt=I[:,[0,3,4,5]]
# y=b0x0+b1x1+b2x2+b3x3
#0->x0-11111         0<0.05           0<0.05   
#3->x3-RnD           0<0.05     x1    0<0.05
#4->x4-Admin         0.6>0.05   x2    0.602>0.5   admin column highly singinificant remove  
#5->x5-Marketing     0.123>0.05 x3    0.105>0.05

Iopt=I[:,[0,3,5]]
# y=b0x0+b1x1+b2x2
#0->x0-11111         0<0.05           0<0.05   
#3->x3-RnD           0<0.05     x1    0<0.05       x1  0.
#5->x5-Marketing     0.123>0.05 x3    0.105>0.05   x2  0.06>0.05
Optimizer=SM.OLS(endog=Dp1,exog=Iopt).fit()
Optimizer.summary()