## BINARY CLASSIFIER
## it is used to classify into two states: true or false

from loading_mnist import LoadData,SplitData
tool0=LoadData()
data=tool0.loadData()

tool1=SplitData(data=data,train_size=60000)
X_train,y_train,X_test,y_test=tool1.splitAndShuffleData()

y_train5=(y_train==5)
y_test5=(y_test==5)

#we will use stochastic gradient descent to train the data
from sklearn.linear_model import SGDClassifier

sgd_clf=SGDClassifier()
sgd_clf.fit(X_train,y_train5)

print(sgd_clf.predict([data['data'][0]]))