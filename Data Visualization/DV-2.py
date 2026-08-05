## CORRELATION
# we can look for relation between different attributes

import pandas as pd
data=pd.read_csv('housing.csv')

corr_matrix=data.corr(numeric_only=True)

print(corr_matrix['median_house_value'].sort_values(ascending=False))