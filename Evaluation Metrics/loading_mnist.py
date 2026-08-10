## MNIST is a very famous data used for testing classification ML models

from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if __name__=="__main__":
    mnist=fetch_openml('mnist_784',version=1,as_frame=False)
    x0=mnist["data"]
    y0=mnist["target"]
    print(x0.shape)
    print(y0.shape)

    random=x0[1]
    random_image=random.reshape(28,28)

    plt.imshow(random_image,cmap=matplotlib.cm.binary,interpolation="nearest")
    plt.show()


class LoadData():
    def __init__(self):
        pass
    def loadData(self):
        return fetch_openml('mnist_784',version=1,as_frame=False,parser='auto')

class SplitData():
    def __init__(self,data,train_size):
        self.data=data
        self.train_size=train_size
    def splitAndShuffleData(self):
        x=self.data["data"]
        y=self.data["target"].astype(np.uint8)
        X_train,Y_train,X_test,Y_test=x[:self.train_size],y[:self.train_size],x[self.train_size:],y[self.train_size:]
        shuffle_index=np.random.permutation(self.train_size)
        X_train,Y_train=X_train[shuffle_index],Y_train[shuffle_index]
        return X_train,Y_train,X_test,Y_test

