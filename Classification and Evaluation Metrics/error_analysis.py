## Error Analysis
## analysing errors in the chosen model to improve

## using confusion matrix
from loading_mnist import LoadData,SplitData
import numpy as np
tool0=LoadData()
data=tool0.loadData()

tool1=SplitData(data=data,train_size=60000)
x_train,y_train,x_test,y_test=tool1.splitData()

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

x_train_scaled=scaler.fit_transform(x_train.astype(np.float64))

from sklearn.linear_model import SGDClassifier
sgd_clf=SGDClassifier()
sgd_clf.fit(x_train,y_train)

from sklearn.model_selection import cross_val_predict
y_train_pred=cross_val_predict(sgd_clf,x_train,y_train,cv=3)

from sklearn.metrics import confusion_matrix
conf_mx=confusion_matrix(y_train,y_train_pred)
print(conf_mx)

import matplotlib.pyplot as plt

row_sums=conf_mx.sum(axis=1,keepdims=True)
norm_conf_mx=conf_mx/row_sums

np.fill_diagonal(norm_conf_mx,0)
plt.matshow(conf_mx,cmap=plt.cm.gray)
plt.show()
