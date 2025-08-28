import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.base_new import DecisionTree
from metrics import *
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay,classification_report
from sklearn.metrics import mean_absolute_error,root_mean_squared_error
#DIDO
N = 30
P = 5
X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(5)})
y = pd.Series(np.random.randint(P, size=N), dtype="category")
print("X:", X)
print("y", y)

des_tr = DecisionTreeClassifier(criterion = 'entropy', max_depth = 5)
des_tr.fit(X,y)

y_p = des_tr.predict(X)
c = confusion_matrix(y,y_p)
ConfusionMatrixDisplay(c).plot(cmap= 'viridis')

print(classification_report(y,y_p))

for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print("Criteria :", criteria)
    print("Accuracy: ", accuracy(y_hat, y))
    
    print("Precision: ", precision(y_hat, y, 1))
    print("Recall: ", recall(y_hat, y, 1))

#DIRO
N = 30
P = 5
X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(5)})
y = pd.Series(np.random.randn(N))
print("X:", X)
print("y", y)

reg_tr = DecisionTreeRegressor(criterion = 'squared_error', max_depth = 5)
reg_tr.fit(X,y)

y_p = reg_tr.predict(X)
print("Mean Absolute Error:", mean_absolute_error(y_p,y))
print("Root Mean Square Error:", root_mean_squared_error(y_p,y))

criteria = "MSE"
tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
tree.fit(X, y)
y_hat = tree.predict(X)
tree.plot()
print("Criteria :", criteria)
print("RMSE: ", rmse(y_hat, y))
print("MAE: ", mae(y_hat, y))
