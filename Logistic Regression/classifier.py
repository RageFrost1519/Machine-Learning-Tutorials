from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

#loading dataset and features
iris=datasets.load_iris()
x=iris['data'][:,3:]
y=(iris['target']==2).astype(np.int16)

#training a logistic regression model
from sklearn.linear_model import LogisticRegression

log_reg=LogisticRegression()
log_reg.fit(x,y)

x_val=np.linspace(0,3,1000).reshape(-1,1)
y_proba=log_reg.predict_proba(x_val)
plt.plot(x_val,y_proba[:,1],'g-',label="Iris-Virginica")
plt.plot(x_val,y_proba[:,0],'r--',label="Not Iris-Virginica")
plt.show()