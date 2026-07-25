#learning curve

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

def plot_learning_curve(model,x,y):
    x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.2)
    train_errors,val_errors=[],[]
    for m in range(1,len(x_train)):
        model.fit(x_train[:m],y_train[:m])
        y_train_predict=model.predict(x_train[:m])
        y_val_predict=model.predict(x_val)
        train_errors.append(mean_squared_error(y_train_predict,y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict,y_val))
    plt.plot(np.sqrt(train_errors),"r-+",linewidth=2,label="train")
    plt.plot(np.sqrt(val_errors),"b-",linewidth=3,label="value") 
    plt.show()

m=1000
x=6*np.random.rand(m,1)-3
y=0.5+x**2+x+np.random.randn(m,1)

lin_reg=LinearRegression()
plot_learning_curve(lin_reg,x,y)
