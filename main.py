import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# prepare model
model = LinearRegression()

# prepare data
X, y = np.arange(10).reshape((5, 2)), range(5)

# split data - train / test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# train model
model.fit(X_train, y_train)

# predict
predictions = model.predict(X_test)

# results
print("Train values:")
print(X_train)
print(y_train)
print("Test: ")
print(X_test)
print("Predicted results: ")
print(predictions)
print("Test results: ")
print(y_test)
