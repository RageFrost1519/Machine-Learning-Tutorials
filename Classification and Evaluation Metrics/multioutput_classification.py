from loading_mnist import LoadData,SplitData
tool0=LoadData()
data=tool0.loadData()

tool1=SplitData(data=data,train_size=60000)
x_train,y_train,x_test,y_test=tool1.splitData()

import numpy as np
import random as rnd

noise_train=np.random.randint(0,100,(len(x_train),784))
noise_test=np.random.randint(0,100,(len(x_test),784))

x_train_mod=x_train+noise_train
x_test_mod=x_test+noise_test

y_train_mod=x_train
y_test_mod=x_test

from sklearn.neighbors import KNeighborsClassifier
knn_clf=KNeighborsClassifier()

knn_clf.fit(x_train_mod,y_train_mod)
clean_digit=knn_clf.predict([x_test_mod[1]])

clean_digit=clean_digit.reshape(-1,28)

original_image=x_test_mod[1].reshape(-1,28)

import matplotlib.pyplot as plt

# Create a figure with 1 row and 2 columns
# figsize=(10, 5) sets the window width to 10 and height to 5 inches
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# --- First Subplot: Original Image ---
axes[0].imshow(original_image, cmap="gray_r")
axes[0].set_title("Original Image")
axes[0].set_xticks([]) # Adding these to keep both boxes clean
axes[0].set_yticks([])

# --- Second Subplot: Cleaned Image ---
axes[1].imshow(clean_digit, cmap="gray_r")
axes[1].set_title("Cleaned Image")
axes[1].set_xticks([])
axes[1].set_yticks([])

# Adjusts spacing so titles don't overlap
plt.tight_layout() 

# Show both plots at once
plt.show()
plt.show()