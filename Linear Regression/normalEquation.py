#the normal equation
#linear regression model: y=a0+a1x1+....anxn

#y=AX A=[a0,a1,...an] X=[1,x1,...xn]

#the normal eqn is a generalized form of the best fitting line of any linear regression model
# A=(X^(T).X)^(-1).X^(T).y 
# where A is the value of parameters that minimizes cost function
# y is the vector of target values containing y(1) to y(m)

#we will implement this using the housing price dataset

import pandas as pd
import numpy as np


#loading training data in numpy arrays
df=pd.read_csv('housing_price_data.csv')
data=pd.read_csv('housing_price_data.csv',usecols=['SquareFeet','Bedrooms','Bathrooms']) 
data=data.to_numpy()
prices=pd.read_csv('housing_price_data.csv',usecols=['Price'])
prices=prices.to_numpy()

x_mean = np.mean(data, axis=0)
x_std = np.std(data, axis=0)
data = (data - x_mean) / x_std

data=np.c_[np.ones((data.shape[0],1)),data] #adding x0=1 to each training set

A=np.linalg.inv(data.T.dot(data)).dot(data.T).dot(prices)
#using A to predict the price of a house

X=np.array([[1,1684,4,3]]) #the first element 1 is added to 
print(A.T.dot(X.T))

