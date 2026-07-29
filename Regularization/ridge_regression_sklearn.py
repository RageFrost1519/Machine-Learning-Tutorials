##Ridge regression using scikit learn module

from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression

x,y=make_regression(n_samples=200,n_features=10,effective_rank=2,noise=15,random_state=42)

ridge_reg=Ridge(alpha=1,solver="cholesky")
ridge_reg.fit(x,y)
