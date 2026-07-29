##Ridge regression using scikit learn module

from sklearn.linear_model import Ridge,LinearRegression
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

x,y=make_regression(n_samples=200,n_features=10,effective_rank=2,noise=15,random_state=42)

ridge_reg=Ridge(alpha=1,solver="cholesky")

#comparing ridge regression and linear regression using cross validation 
#the data is chosen in such a way that linear regression does not perform good

lin_reg=LinearRegression()

score_liner=-cross_val_score(lin_reg,x,y.ravel(),scoring='neg_mean_squared_error',cv=5).mean()
score_ridge=-cross_val_score(ridge_reg,x,y.ravel(),scoring='neg_mean_squared_error',cv=5).mean()
print(f'Linear Model Score : {score_liner}')
print(f'Ridge Model Score : {score_ridge}')
