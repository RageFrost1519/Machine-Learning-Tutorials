## MNIST is a very famous data used for testing classification ML models

from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

mnist=fetch_openml('mnist_784',version=1,as_frame=False)
x=mnist["data"]
y=mnist["target"]
print(x.shape)
print(y.shape)

random=x[1]
random_image=random.reshape(28,28)

plt.imshow(random_image,cmap=matplotlib.cm.binary,interpolation="nearest")
plt.show()

class LoadData():
    def __init__(self):
        pass
    def loadData():
        return fetch_openml('mnist_784',version=1,as_frame=False)

class SplitData():
    def __init__(self,data,train_size):
        self.data=data
        self.train_size=train_size
    def splitAndShuffleData(self):
        x=self.data["data"]
        y=self.data["target"]
        X_train,Y_train,X_test,Y_test=x[:self.train_size],y[:self.train_size],x[self.train_size:],y[self.train_size:]
        shuffle_index=np.random.permutation(self.train_size)
        X_train,Y_train=X_train[shuffle_index],Y_train[shuffle_index]
        return X_train,Y_train,X_test,Y_test

