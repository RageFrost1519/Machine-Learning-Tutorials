## data cleaning
## some values in the data might be missing or not available 

## we can fix this by dropping that instance using .dropna()
## or we can drop that whole attribute using .drop()
## or we can fill in those missing values with values like mean and median of the data

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

## we will replace the missing values by the median

median=housing["total_bedrooms"].median()
housing["total_bedrooms"]=housing["total_bedrooms"].fillna(median)

## now this data can be used to train the model

print(strat_train_set.info())
print(housing.info())