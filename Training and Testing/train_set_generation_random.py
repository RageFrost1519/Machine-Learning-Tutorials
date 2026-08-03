## TRAINING SET GENERATION
# creating a good data set for training is really important for good training of the model
# the data used for training shouldnt be random but should represent the whole data set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('housing.csv')

## in this file, we will implement a test data generating code which randomly selects test data


data['income_cat']=np.ceil(data['median_income']/1.5)
data['income_cat'].where(data['income_cat']<5,5.0,inplace=True)

def split_train_test(data,test_ratio):
    shuffled_indices=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled_indices[:test_set_size]
    train_indices=shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

train_set,test_set=split_train_test(data,0.2)

## to make sure the same data is used for training we will assign an identifier to each row and use hashing to make sure same data is selected

import hashlib

def test_set_check(identifier,test_ratio,hash):
    return hash(np.int64(identifier)).digest()[-1]<256*test_ratio

def split_train_test_by_id(data,test_ratio,id_column,hash=hashlib.md5):
    ids=data[id_column]
    in_test_set=ids.apply(lambda id_: test_set_check(id_,test_ratio,hash))
    return data.loc[~in_test_set],data.loc[in_test_set]
    ## the first set is the training set and second set is the testing set

data_with_id=data.reset_index()
train_set0,test_set0=split_train_test_by_id(data_with_id,0.2,"index")

print(data['income_cat'].value_counts()/len(data))
print(train_set['income_cat'].value_counts()/len(train_set))