# this is simple linear regression basic programe
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

#read dataset
dataframe = pd.read_csv('challenge_dataset.txt', sep=",", header=None, names=["key", "value"])
x_frame = dataframe[['key']]
y_frame = dataframe[['value']]

x_mean = np.mean(x_frame, dtype=float)
y_mean = np.mean(y_frame, dtype=float)

a = np.sum(np.multiply(x_frame-x_mean, y_frame-y_mean))/np.sum((x_frame - x_mean)**2)
b = np.subtract(y_mean , a*x_mean)
y_predict = a*x_frame + b['value']

RMSE = np.sqrt(np.sum(np.subtract(y_predict,y_frame)**2)/len(y_frame))

print RMSE

plt.scatter(x_frame, y_frame)
plt.plot(x_frame, y_predict)

plt.show()