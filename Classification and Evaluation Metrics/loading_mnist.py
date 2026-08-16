import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import urllib.request
import os

# We no longer need fetch_openml from sklearn

class LoadData():
    def __init__(self):
        # We will use the highly compressed, fast-downloading version from Google/Keras
        self.url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        self.file_name = "mnist.npz"

    def loadData(self):
        # 1. Download the file only if it doesn't already exist locally
        if not os.path.exists(self.file_name):
            print("Downloading MNIST dataset (11MB)...")
            urllib.request.urlretrieve(self.url, self.file_name)
            print("Download complete!")
        
        # 2. Load the local .npz file
        with np.load(self.file_name, allow_pickle=True) as f:
            x_train, y_train = f['x_train'], f['y_train']
            x_test, y_test = f['x_test'], f['y_test']
        
        # 3. Combine and reshape to match the old OpenML format (70000 images, 784 pixels each)
        x_combined = np.concatenate((x_train, x_test)).reshape(70000, 784)
        y_combined = np.concatenate((y_train, y_test))
        
        return {"data": x_combined, "target": y_combined}

class SplitData():
    def __init__(self, data, train_size):
        self.data = data
        self.train_size = train_size
        
    def splitAndShuffleData(self):
        x = self.data["data"]
        # The .npz file already stores labels as integers, so we just ensure it's uint8
        y = self.data["target"].astype(np.uint8)
        
        X_train, Y_train = x[:self.train_size], y[:self.train_size]
        X_test, Y_test = x[self.train_size:], y[self.train_size:]
        
        shuffle_index = np.random.permutation(self.train_size)
        X_train, Y_train = X_train[shuffle_index], Y_train[shuffle_index]
        
        return X_train, Y_train, X_test, Y_test
    
    def splitData(self):
        x = self.data["data"]
        # The .npz file already stores labels as integers, so we just ensure it's uint8
        y = self.data["target"].astype(np.uint8)
                
        X_train, Y_train = x[:self.train_size], y[:self.train_size]
        X_test, Y_test = x[self.train_size:], y[self.train_size:]

        return X_train, Y_train, X_test, Y_test

if __name__ == "__main__":
    # Test our new, much faster local loader
    tool = LoadData()
    mnist = tool.loadData()
    
    x0 = mnist["data"]
    y0 = mnist["target"]
    print("Images shape:", x0.shape)
    print("Labels shape:", y0.shape)

    random_image = x0[1].reshape(28, 28)

    plt.imshow(random_image, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.show()
