## data cleaning
## it can also be implemented using sklearn's imputer class

import pandas as pd
import numpy as np
data=pd.read_csv('housing.csv')

## we will use the stratified data

data['income_cat']=np.ceil(data['median_income']/1.5)
data['income_cat'].where(data['income_cat']<5,5.0,inplace=True)

from sklearn.model_selection import StratifiedShuffleSplit

split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index,test_index in split.split(data,data['income_cat']):
    strat_train_set=data.loc[train_index]
    strat_test_set=data.loc[test_index]

## it is good to separate the input features from output labels during data cleaning

housing=strat_train_set.drop("median_house_value",axis=1)
housing_labels=strat_train_set["median_house_value"].copy()

## we will create an imputer instance first
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy="median")

## we will remove the ocean proximity columnn sicne it does not have numerical values
housing_num=housing.drop("ocean_proximity",axis=1)
imputer.fit(housing_num)

x=imputer.transform(housing_num)
housing_tr=pd.DataFrame(x,columns=housing_num.columns)

print(housing_tr.info())