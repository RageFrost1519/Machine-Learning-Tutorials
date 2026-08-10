## implementation of cross validation without sklearn

from loading_mnist import LoadData,SplitData
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import SGDClassifier
from sklearn.base import clone

tool0=LoadData()
data=tool0.loadData()

tool1=SplitData(data=data,train_size=60000)
X_train,y_train,X_test,y_test=tool1.splitAndShuffleData()

y_train5=(y_train==5)
y_test5=(y_test==5)

skfolds=StratifiedKFold(n_splits=3)

sgd_clf=SGDClassifier()
sgd_clf.fit(X_train,y_train5)


for train_index,test_index in skfolds.split(X_train,y_train5):
    clone_clf=clone(sgd_clf)
    X_train_folds=X_train[train_index]
    y_train_folds=y_train5[train_index]
    X_test_folds=X_train[test_index]
    y_test_folds=y_train5[test_index]

    clone_clf.fit(X_train_folds,y_train_folds)
    y_pred=clone_clf.predict(X_test_folds)
    n_correct=sum(y_test_folds==y_pred)
    print(n_correct/len(y_pred))


