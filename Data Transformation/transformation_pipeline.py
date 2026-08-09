## PIPELINES
# we can create a pipeline to apply all the transformation methods to a dataset at once

import numpy as np
import pandas as pd
data=pd.read_csv('housing.csv')


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from custom_transformers import CombinedAttributesAdder

num_pipeline=([    ##transformations to apply on only numerical values
    ('imputer',SimpleImputer(strategy="median"))
    ('attribs_adder',CombinedAttributesAdder())
    ('std_scaler',StandardScaler())
])