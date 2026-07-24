#stochastic gradient descent
#instead of using the whole training set, only one example is used for each iteration
#this can cause the cost function to bounce but on avg the cost funcc decreases.

#learning rate is gradually reduced to escape local minima quickly but then to settle at global minimum
#learning rate is decided using learning schedulee

t0,t1=5,50
n_epochs=50

def learning_schedule(t):
    return t0/(t+t1)

import pandas as pd
import numpy as np
import time

df=pd.read_csv('housing_price_data - Copy.csv')
x=pd.read_csv('housing_price_data - Copy.csv',usecols=['SquareFeet','Bedrooms','Bathrooms']) 
x=x.to_numpy()
y=pd.read_csv('housing_price_data - Copy.csv',usecols=['Price'])
y=y.to_numpy().flatten()

m=x.shape[0]

x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
x = np.c_[np.ones((m, 1)), x]

A=np.zeros(x.shape[1])
start_time=time.time()

for epoch in range(n_epochs):
    for i in range(m):
        random_index=np.random.randint(m)
        xi=x[random_index:random_index+1]
        yi=y[random_index:random_index+1]
        gradients=2*xi.T.dot(xi.dot(A)-yi)
        eta=learning_schedule(epoch*m+i)
        A=A-eta*gradients

print(time.time()-start_time)
