## instead of taking random instances from data to train the model
## we take data instances which can represent the whole dataset

## here we will implement code to select data which can choose a training dataset based on median income of households

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv('housing.csv')
# print(data.describe())

#median income is in the range 0.04999-15.0001 (10k dollars)
#we will divide it by 1.5 and apply ceil function to get a discrete range from 1-10
#then we choose instances to create a training dataset which can represent the whole data based on median income distribution

data.hist(bins=20,figsize=(20,15))

data['income_cat']=np.ceil(data['median_income']/1.5)
data['income_cat'].where(data['income_cat']<5,5.0,inplace=True)

#since no of households with median income greater than 7 are not much we will create only 5 categories

from sklearn.model_selection import StratifiedShuffleSplit

split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index,test_index in split.split(data,data['income_cat']):
    strat_train_set=data.loc[train_index]
    strat_test_set=data.loc[test_index]

print(data['income_cat'].value_counts()/len(data))
print(strat_train_set['income_cat'].value_counts()/len(strat_train_set))