import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

print("Number of classes: ", len(np.unique(iris_y)))
print("Number of data points: ", len(iris_y))

X0 = iris_X[iris_y == 0]
X1 = iris_X[iris_y == 1]
X2 = iris_X[iris_y == 2]
# print("Sample from X0: \n", X0[:5])
# print("Sample from X1: \n", X1[:5])
# print("Sample from X2: \n", X2[:5])

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=50)

model = neighbors.KNeighborsClassifier(n_neighbors=10, p = 2, weights="distance")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(y_pred[:40])
print(y_test[:40])

print("MAE: ", mean_absolute_error(y_test, y_pred))
print("Accuracy score: ", accuracy_score(y_test, y_pred))