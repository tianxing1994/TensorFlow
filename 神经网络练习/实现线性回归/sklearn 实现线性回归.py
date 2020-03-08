import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.linear_model import LinearRegression

x = np.linspace(0,10,20) + np.random.rand(20) * 2
y = np.linspace(0,10,20) + np.random.rand(20) * 2

linear = LinearRegression()
linear.fit(x.reshape(-1,1),y)

X = np.linspace(2,10,100)
y_ = linear.predict(X.reshape(-1,1))

# y = ax + b
# print(linear.coef_, linear.intercept_)
# [1.05541833] -0.41988828537867917

plt.plot(X,y_,c="r")
plt.scatter(x,y)
plt.show()