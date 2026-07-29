##Ridge Regression
#it is a regularization method used to deal with overfitting
#a little bias is added to the curve by adding a penalty to the cost function which causes a significant drop in the variance

#Like linear regression it also has a closed form
# A=(X.T.dot(X)+alpha*I)^(-1)X.T.y

import numpy as np

class RidgeRegressin():
    def __init__(self,alpha=1.0):
        self.alpha=alpha
        self.weights=None

    def fit(self,x,y):
        m=x.shape[0]
        x1=np.c_[np.ones((m,1)),x]
        n=x1.shape[1]
        I=np.eye(n)
        I[0,0]=0  #we put first diagonal element as 0 because the y-intercept is not influenced by ridge regression
        A=x1.T.dot(x1)+self.alpha*I
        b=x1.T.dot(y)
        self.weights=np.linalg.inv(A).dot(b)

    def predict(self,X):
        if self.weights is None:
            raise ValueError("Model has not been fitted yet.")

        m=X.shape[0]
        x=np.c_[np.ones(m,1),X]
        return x.dot(self.weights)

    