## we will plot the training data on a graph
## since it is a housing price data we will plot based on location

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv('housing.csv')
data['income_cat']=np.ceil(data['median_income']/1.5)
data['income_cat'].where(data['income_cat']<5,5.0,inplace=True)

from sklearn.model_selection import StratifiedShuffleSplit

split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index,test_index in split.split(data,data['income_cat']):
    strat_train_set=data.loc[train_index]
    strat_test_set=data.loc[test_index]

housing=strat_train_set.copy()

"'housing.plot(kind='scatter',x='longitude',y='latitude',alpha=0.1)'"

## using alpha=0.1 highlights the places with more data points

## we can use more detailed graphs to represent price and population.
## population is represented by size and price is represented by colour


housing.plot(kind='scatter',x='longitude',y='latitude',alpha=0.4,s=housing['population']/100,
             label="POPULATION",c='median_house_value',cmap=plt.get_cmap('jet'),colorbar=True)
plt.show()