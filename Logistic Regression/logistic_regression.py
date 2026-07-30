import numpy as np

class LogisticRegression:
    def __init__(self,learning_rate=0.01,num_iterations=1000):
        self.learning_rate=learning_rate
        self.num_iterations=num_iterations
        self.weights=None
        self.bias=None

    def sigmoid(self,z):
        ##it takes a real value and maps it to a value bw 0 and 1
        ##np.clip() puts a value in a range.
        ##anything higher becomes upper limit and anything lower becomes lower limit
        z=np.clip(z,-250,250)
        return 1/1+np.exp(-z)

    def fit(self,x,y):
        m,n=x.shape
        self.weights=np.zeros(n)
        self.bias=0

        for i in range(m):
            linear_model=np.dot(x,self.weights)+self.bias
            y_cap=self.sigmoid(linear_model)
            dw=1/m*(np.dot(x.T,y_cap))
            db=1/m*(np.sum(y_cap-y))
            self.weights-=self.learning_rate*dw
            self.bias-=self.learning_rate*db

    def predict_proba(self,x):
        linear_model=np.dot(x,self.weights)+self.bias
        return self.sigmoid(linear_model)

    def predict(self,x,threshold=0.5):
        probabilities=self.predict_proba(x)
        return np.array([1 if p>=threshold else 0 for p in probabilities])