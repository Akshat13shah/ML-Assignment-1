import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

# Reading the data
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])
#print(data)

data["horsepower"] = pd.to_numeric(data["horsepower"], errors = "coerce")
print(data)

data_1 = data.dropna(axis = 0, ignore_index=True)
print(data_1)
#print(data_1.columns.unique())
print(data_1.loc[29:37])
print(type(data_1.loc[32]["horsepower"]))
for i in data.columns:
       print(i,data[i].dtype)
print(len(pd.unique(data_1["car name"])))
X = data_1.drop("car name", axis= 1)
y = data_1["car name"]


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay,classification_report

des_tr = DecisionTreeClassifier(criterion = 'entropy', max_depth = 5)
des_tr.fit(X,y)

y_p = des_tr.predict(X)
c = confusion_matrix(y,y_p)
ConfusionMatrixDisplay(c).plot(cmap= 'viridis')

print(classification_report(y,y_p))

criteria = "entropy"
tree = DecisionTree(criterion=criteria) 
tree.fit(X, y)
y_hat = tree.predict(X)
tree.plot()
print("Criteria :", criteria)
print("Accuracy: ", accuracy(y_hat, y))
print("Precision: ", precision(y_hat, y, cls = "macro"))
print("Recall: ", recall(y_hat, y, cls = "macro"))

# Clean the above data by removing redundant columns and rows with junk values
# Compare the performance of your model with the decision tree module from scikit learn
