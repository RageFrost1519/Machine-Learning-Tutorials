## implementing random training set generation using sklearn

from sklearn.model_selection import train_test_split
import pandas as pd

data=pd.read_csv('housing.csv')

train_set,test_set=train_test_split(data,test_size=0.2,random_state=42)
print(train_set)
print(test_set)