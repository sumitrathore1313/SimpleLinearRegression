import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

#read dataset
dataframe = pd.read_csv('test.txt', sep=",", header=None, names=["key", "value"])
x_frame = dataframe[['key']]
y_frame = dataframe[['value']]


reg = linear_model.LinearRegression()
reg.fit(x_frame, y_frame)

plt.scatter(x_frame, y_frame)
plt.plot(x_frame, reg.predict(x_frame))
plt.show()

error = reg.predict(x_frame)
error = np.squeeze(error)

loss = y_frame['value'] - error

total_loss = loss.sum()

print total_loss
