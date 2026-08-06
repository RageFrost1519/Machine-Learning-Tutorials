## transforming attributes
## in a data we might want to change a attribute for better performance

# eg. total rooms in a household is more helpful than total rooms in a district

import pandas as pd
data=pd.read_csv('housing.csv')

data['rooms_per_household']=data['total_rooms']/data['households']
data['bedrooms_per_room']=data['total_bedrooms']/data['total_rooms']
data['population_per_household']=data['population']/data['households']

corr_matrix=data.corr(numeric_only=True);
print(corr_matrix['median_house_value'].sort_values(ascending=False))

