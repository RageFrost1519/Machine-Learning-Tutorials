## Confusion Matrix
from loading_mnist import LoadData,SplitData
tool0=LoadData()
data=tool0.loadData()

tool1=SplitData(data=data,train_size=60000)
X_train,y_train,X_test,y_test=tool1.splitAndShuffleData()

from sklearn.model_selection import cross_val_predict
## cross_val_predict performs k-fold cross validation and returns the predictions made on each fold

y_train5=(y_train==5)
y_test5=(y_test==5)

from sklearn.linear_model import SGDClassifier

sgd_clf=SGDClassifier()
sgd_clf.fit(X_train,y_train5)

from sklearn.model_selection import cross_val_predict
y_train_pred=cross_val_predict(sgd_clf,X_train,y_train5,cv=3)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_train5,y_train_pred))  ## confusion matrix takes the actual values first then the predictions

## OUTPUT 
# [[53850   729]
#  [ 1481  3940]]

## columns represent actual values and rows represent predicted values
## [0][0] represents true negatives (not 5s that were correctly classified as not 5)
## [0][1] represent false positives (not 5s that were incorrectly classified as 5)
## [1][0] represent false negatives (5s that were incorrectly classified as not 5)
## [1][1] represent true positives (5s that were correctly classified as 5)

from sklearn.metrics import precision_score,recall_score
print(precision_score(y_train5,y_train_pred))
print(recall_score(y_train5,y_train_pred))

from sklearn.metrics import f1_score
print(f1_score(y_train5,y_train_pred))