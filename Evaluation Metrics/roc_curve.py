## Receiver operating characterstic
## it plots true posititve rate (recall) against false positive rate
## fpr=1-tnr (specificity)
## thus roc curve plots sensitivitu (recall) vs specificity

from loading_mnist import LoadData,SplitData
tool0=LoadData()
data=tool0.loadData()

tool1=SplitData(data=data,train_size=60000)
X_train,y_train,X_test,y_test=tool1.splitAndShuffleData()

y_train5=(y_train==5)
y_test5=(y_test==5)

from sklearn.linear_model import SGDClassifier

sgd_clf=SGDClassifier()
sgd_clf.fit(X_train,y_train5)

from sklearn.model_selection import cross_val_predict
y_scores=cross_val_predict(sgd_clf,X_train,y_train5,cv=3,method="decision_function")

from sklearn.metrics import roc_curve
fpr,tpr,thresholds=roc_curve(y_train5,y_scores)

import matplotlib.pyplot as plt

def plot_roc_curve(fpr,tpr,label=None):
    plt.plot(fpr,tpr,linewidth=2,label=label)
    plt.plot([0,1],[0,1],'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

plot_roc_curve(fpr,tpr)
plt.show()

## to compare different ROCs we use area under the curve as a measure of difference.
## higher the area better the ROC

## use precision/recall curve when we care more about false positives or positive class is rare

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_train5,y_scores))