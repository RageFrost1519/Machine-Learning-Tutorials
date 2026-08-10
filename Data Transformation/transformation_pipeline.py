## PIPELINES
# we can create a pipeline to apply all the transformation methods to a dataset at once

import numpy as np
import pandas as pd
data=pd.read_csv('housing.csv')


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,LabelBinarizer,OneHotEncoder
from sklearn.impute import SimpleImputer
from custom_transformers import CombinedAttributesAdder,DataFrameSelector

housing_num=data.drop("ocean_proximity",axis=1)

num_attributes=list(housing_num)
cat_attributes=["ocean_proximity"]

num_pipeline=Pipeline([    ##transformations to apply on only numerical values
    ('selector',DataFrameSelector(num_attributes)),
    ('imputer',SimpleImputer(strategy="median")),
    ('attribs_adder',CombinedAttributesAdder()),
    ('std_scaler',StandardScaler()),
])

cat_pipeline=Pipeline([
    ('selector',DataFrameSelector(cat_attributes)),
    ('label_binarizer',OneHotEncoder(sparse_output=False)),
])

from sklearn.pipeline import FeatureUnion

full_pipeline=FeatureUnion([
    ('num_pipeline',num_pipeline),
    ('cat_pipeline',cat_pipeline),
])

data_transformed=full_pipeline.fit_transform(data)
print(data_transformed)
