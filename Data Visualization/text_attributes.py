## text attributes can also be used for data prediction
## one way is to assign each category a number

import pandas as pd
import numpy as np
data=pd.read_csv("housing.csv")

## we will apply this on the ocean proximity category
# print(data['ocean_proximity'].value_counts()) 

# there are 5 categories each will get a number from 0 to 4

from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
housing_cat=data['ocean_proximity']
housing_cat_encoded=encoder.fit_transform(housing_cat)

#* print(encoder.classes_)
#* print(housing_cat_encoded)

# returns an array with a label for each entry in the dataset based on the ocean proximity value

# this technique has a major drawback
# two english words which are closely related in meaning may get alloted numerical values with very high difference
# this results in difficulty in finding a good fitting curve for the data 

# another way is to use one-hot vectors
# it is a n x m matrix (n-no of entries in dataset,m-no of categories)
# for each entry in dataset the corresponding category attribute is 1 and rest all are 0

# since it will have a lot of 0s and very less 1s we use a sparse matrix
# a sparse matrix stores the location of nonzero elements instead of the actual matrix

from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder()
housing_cat_1hot=encoder.fit_transform(housing_cat_encoded.reshape(-1,1))

#* print(housing_cat_1hot.toarray())

# we can apply text to integer and integer to 1 hot at once using labelBinarizer

from sklearn.preprocessing import LabelBinarizer
encoder=LabelBinarizer()
housing_cat_1hot_direct=encoder.fit_transform(housing_cat)
print(housing_cat_1hot_direct)