# polynomial regression
# instead of fitting a straight line we can fit a curve using powers of features
# implementation using scikit learn library

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

m=100
x=6*np.random.rand(m,1)-3
y=0.5+0.5*x**2+x+np.random.randn(m,1)

poly_features=PolynomialFeatures(degree=2,include_bias=False)
x_poly=poly_features.fit_transform(x)

lin_reg=LinearRegression()
lin_reg.fit(x_poly,y)
print(lin_reg.intercept_,lin_reg.coef_)