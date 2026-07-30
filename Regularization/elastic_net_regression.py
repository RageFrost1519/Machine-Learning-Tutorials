##ELASTIC NET REGRESSION
# it combines both ridge and lasso regression

from sklearn.linear_model import ElasticNet,LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score

x,y=make_regression(n_samples=1000,n_features=10,effective_rank=2,noise=15,random_state=42)

alphas = [0.001, 0.01, 0.1, 1, 10, 100]
lin_reg=LinearRegression()

i=1
for a in alphas:
    elastic_net=ElasticNet(alpha=a,l1_ratio=0.5)
    score_liner=-cross_val_score(lin_reg,x,y.ravel(),scoring='neg_mean_squared_error',cv=5).mean()
    score_lasso=-cross_val_score(elastic_net,x,y.ravel(),scoring='neg_mean_squared_error',cv=5).mean()
    print(f"------ITERATION {i}------")
    print(f'Linear Model Score : {score_liner}')
    print(f'Elastic Net Model Score : {score_lasso}')
    i+=1