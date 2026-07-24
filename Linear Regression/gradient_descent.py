#gradient descent
#this is another way of finding the parameters that fits the curve best

#we calculate the derivative of cost function wrt each parameter and take small steps towards minima until the values converge
#as we are using numpy, we can easily do this using
#gradient_vector=(2/m)X^(T)(XA-y)
#A=A-eta*gradient_vector

#we will load the dataset first

import pandas as pd
import numpy as np
import time

df=pd.read_csv('housing_price_data - Copy.csv')
x=pd.read_csv('housing_price_data - Copy.csv',usecols=['SquareFeet','Bedrooms','Bathrooms']) 
x=x.to_numpy()
y=pd.read_csv('housing_price_data - Copy.csv',usecols=['Price'])
y=y.to_numpy().flatten()

x_mean = np.mean(x, axis=0)
x_std = np.std(x, axis=0)
x = (x - x_mean) / x_std

x=np.c_[np.ones((x.shape[0],1)),x]

eta=0.01    #learning rate
n_iterations=10000
m=x.shape[0]


A=np.zeros(x.shape[1])   #all parameters initialized to 0
start_time=time.time()


for i in range(n_iterations):
    g=(2/m)*(x.T)@(x@A-y)
    A=A-eta*g

print(time.time()-start_time)

#this is batch gradient descent as it uses the whole training set to compute parameters