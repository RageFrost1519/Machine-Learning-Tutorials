## CORRELATION
# we can look for relation between different attributes

import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('housing.csv')

corr_matrix=data.corr(numeric_only=True)

print(corr_matrix['median_house_value'].sort_values(ascending=False))

## another way to check correlation is using panda's scatter matrix function
## it plots every attribute against every other attribute

from pandas.plotting import scatter_matrix

attributes=["median_house_value","median_income","total_rooms","housing_median_age"]

#scatter_matrix(data[attributes],figsize=(12,8))


## we can also plot only 2 attributes using plot function 

data.plot(kind="scatter",x='median_income',y='median_house_value',alpha=0.1)
plt.show()