#cross validation
#it is a method to check which ml method works best for the given data

#the whole data is divided into k folds and some folds are used for training and others for testing
#cross validation uses all the blocks for testing one at a time

#we will use cross validation to see which model is good for a curved data

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

#generating data
np.random.seed(42)
X=np.sort(np.random.rand(50,1)*2*np.pi,axis=0)
y=np.sin(X).ravel()+np.random.randn(50)*0.3

#generating models
model_linear=LinearRegression()
model_poly3=make_pipeline(PolynomialFeatures(3),LinearRegression())
model_poly15=make_pipeline(PolynomialFeatures(15),LinearRegression())

# Score the Linear Model
scores_linear = -cross_val_score(model_linear, X, y, scoring='neg_mean_squared_error', cv=5)
mse_linear = scores_linear.mean()

# Score the Degree 3 Polynomial
scores_poly3 = -cross_val_score(model_poly3, X, y, scoring='neg_mean_squared_error', cv=5)
mse_poly3 = scores_poly3.mean()

# Score the Degree 15 Polynomial
scores_poly15 = -cross_val_score(model_poly15, X, y, scoring='neg_mean_squared_error', cv=5)
mse_poly15 = scores_poly15.mean()

# 4. Print the results
print("Cross-Validation Mean Squared Errors (Lower is better):")
print("-" * 55)
print(f"Linear Model (Degree 1):  {mse_linear:.3f} (High Bias / Underfitting)")
print(f"Polynomial (Degree 3):    {mse_poly3:.3f} (The Winner!)")
print(f"Polynomial (Degree 15):   {mse_poly15:.3f} (High Variance / Overfitting)")

# 5. Visualize the results for intuition
X_plot = np.linspace(0, 2 * np.pi, 100)[:, np.newaxis]

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='black', label='Data', zorder=5)

# Fit models on the full dataset just for drawing the lines
model_linear.fit(X, y)
model_poly3.fit(X, y)
model_poly15.fit(X, y)

plt.plot(X_plot, model_linear.predict(X_plot), color='blue', label=f'Linear (MSE: {mse_linear:.2f})')
plt.plot(X_plot, model_poly3.predict(X_plot), color='green', label=f'Degree 3 (MSE: {mse_poly3:.2f})')
plt.plot(X_plot, model_poly15.predict(X_plot), color='red', label=f'Degree 15 (MSE: {mse_poly15:.2f})')

plt.title("Cross-Validation: Finding the Best Model")
plt.xlabel("X")
plt.ylabel("y")
plt.ylim(-2.5, 2.5)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()