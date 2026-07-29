# this is the implementation of lasso regression using scikit learn library

from sklearn.linear_model import Lasso,LinearRegression
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

x,y=make_regression(n_samples=1000,n_features=10,effective_rank=2,noise=15,random_state=42)

#comparing ridge regression and linear regression using cross validation 
#the data is chosen in such a way that linear regression does not perform good

lin_reg=LinearRegression()

alphas = [0.005, 0.06, 0.1, 1, 10, 100]
i=1

for a in alphas:
    lasso_reg=Lasso(alpha=a)
    score_liner=-cross_val_score(lin_reg,x,y.ravel(),scoring='neg_mean_squared_error',cv=5).mean()
    score_lasso=-cross_val_score(lasso_reg,x,y.ravel(),scoring='neg_mean_squared_error',cv=5).mean()
    print(f"------ITERATION {i}------")
    print(f'Linear Model Score : {score_liner}')
    print(f'Lasso Model Score : {score_lasso}')
    i+=1
