#bias variance tradeoff

#a biased model is too rigid and makes strong assumptions and ignores the patterns
#eg. a linear regression model assumes a straight line fits the data well 

#a model with high variance is too flexible. it memorizes the random noises in the data.
#because of this it performs very well on training data but terrible on testing data

#a straight line on a curved data will have high bias
#it can be reduced by adding polynomial featues

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline 

#creating data with curves
np.random.seed(42)
x=np.sort(np.random.rand(20,1)*10,axis=0)
y=np.sin(x).ravel()+np.random.randn(20)*0.3

#creating a linear regression model which will have high bias
model_bias=LinearRegression()
model_bias.fit(x,y)

#creating a model with high variance
model_variance=make_pipeline(PolynomialFeatures(15),LinearRegression())
model_variance.fit(x,y)

#plotting the data
# Generate points for a smooth curve
x_plot = np.linspace(0, 10, 300).reshape(-1, 1)

# Predictions
y_bias = model_bias.predict(x_plot)
y_variance = model_variance.predict(x_plot)

# Plot
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# High Bias
ax[0].scatter(x, y, color='black')
ax[0].plot(x_plot, y_bias, color='blue', linewidth=2)
ax[0].set_title("High Bias (Underfitting)")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].grid(True)

# High Variance
ax[1].scatter(x, y, color='black')
ax[1].plot(x_plot, y_variance, color='red', linewidth=2)
ax[1].set_title("High Variance (Overfitting)")
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
ax[1].grid(True)

plt.tight_layout()
plt.show()
