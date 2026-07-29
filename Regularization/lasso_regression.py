## LASSO REGRESSION
# another method of regularization
# only difference from ridge is that we use l1 penalty instead of l2

import numpy as np

#in this implementation we will use gradient descent instead of any closed form

class LassoRegression:
    def __init__(self,learning_rate=0.01,epochs=1000,alpha=1.0):
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.alpha=alpha

    def fit(self,x,y):
        m,n=x.shape
        self.weights=np.zeros(n)
        self.bias=0
        for i in range(self.epochs):
            y_pred=np.dot(x,self.weights)+self.bias
            error=y_pred-y
            dw=(1/m)*np.dot(x.T,error)
            dw+=self.alpha*np.sign(self.weights)
            db=(1/m)*np.sum(error)
            self.weights-=self.learning_rate*dw
            self.bias-=self.learning_rate*db

    def predict(self,x):
        return np.dot(x,self.weights)+self.bias