## Precision Recall Tradeoff
## if you increase the tradeoff to classify something as true
## precision increases but the recall goes down

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

## for each instance, sgd classifier assigns it a score
## based on an appropriate cutoff something is classified as true or false

y_scores=sgd_clf.decision_function([data['data'][0]])

## we will try raising the threshold
threshold=10000
#* print(y_scores>threshold)

## we can choose an appropriate threshold based on our need for precision and recall

from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
y_scores=cross_val_predict(sgd_clf,X_train,y_train5,cv=3,method="decision_function")

from sklearn.metrics import precision_recall_curve
precisions,recalls,thresholds=precision_recall_curve(y_train5,y_scores)

def plot_precision_recall_vs_threshold(precisions,recall,thresholds):
    plt.plot(thresholds,precisions[:-1],"b--",label="Precision")
    plt.plot(thresholds,recall[:-1],'g-',label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0,1])

plot_precision_recall_vs_threshold(precisions,recalls,thresholds)
plt.show()