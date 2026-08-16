## Multiclass Classification
## used to predict more than one class
## two strategies - OneVsOne and OneVsAll

## OneVsAll - a binary classifier is trained for each category
## The category with highest score is the final prediction

## OneVsOne - a binary classifier is trained for each pair of category i.e. there are n(n-1)/2 classifiers
## The category which wins the most contests is the final prediction

## OneVsOne is a better strategy for models which scale poorly with increasing dataset sizes

from loading_mnist import LoadData,SplitData
tool0=LoadData()
data=tool0.loadData()

tool1=SplitData(data,10000)
x_train,y_train,x_test,y_test=tool1.splitData()

from sklearn.linear_model import SGDClassifier
sgd_clf=SGDClassifier()
sgd_clf.fit(x_train,y_train)

print(sgd_clf.predict([x_train[0]]))

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)

#! print(sgd_clf.decision_function([x_train[0]]))  
## returns the score for each category 
## the category with highest score is predicted


## we can also convert sgd classifier from one vs all to one vs one
from sklearn.multiclass import OneVsOneClassifier
ovo_clf=OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(x_train,y_train)
print(ovo_clf.predict([x_train[0]]))

from sklearn.ensemble import RandomForestClassifier
forest_clf=RandomForestClassifier()
forest_clf.fit(x_train,y_train)

print(forest_clf.predict([x_train[0]]))
print(forest_clf.predict_proba([x_train[0]]))