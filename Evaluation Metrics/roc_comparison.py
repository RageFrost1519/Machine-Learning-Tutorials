## comparison of SGDClassifier and RandomForectClassifier by roc curve

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

from sklearn.ensemble import RandomForestClassifier

forest_clf=RandomForestClassifier()
y_probas_forest=cross_val_predict(forest_clf,X_train,y_train5,cv=3,method="predict_proba")

y_scores_forest=y_probas_forest[:,1] ## the second column stores the probability of being in positive class

from sklearn.metrics import roc_curve
fpr_forest,tpr_forest,threshold_forest=roc_curve(y_train5,y_scores_forest)
fpr,tpr,thresholds=roc_curve(y_train5,y_scores)

def plot_roc_curve(fpr,tpr,label=None):
    plt.plot(fpr,tpr,linewidth=2,label=label)
    plt.plot([0,1],[0,1],'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


import matplotlib.pyplot as plt
plt.plot(fpr,tpr,'b:',label="SGD")
plot_roc_curve(fpr_forest,tpr_forest,"Random Forest")
plt.legend(loc="lower right")
plt.show()
