"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)


@dataclass
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature        # index of feature
        self.threshold = threshold    # split threshold
        self.left = left              # left subtree
        self.right = right            # right subtree
        self.value = value            # leaf value (class label or regression mean)

    def is_leaf(self):
        return self.value is not None # here not none means it's a leaf node


class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None
        self.task_type = None  # "classification" or "regression"   

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        # determine task type from y
        self.task_type = "regression" if check_ifreal(y) else "classification"

        # If regression, force mse criterion
        if self.task_type == "regression":
            self.criterion = "mse"
        else:
            # map older naming if user passed them
            if self.criterion in ["information_gain", "information gain"]:
                self.criterion = "entropy"
            if self.criterion in ["gini_index"]:
                self.criterion = "gini"
        self.root = self._build_tree(X, y, depth=0, features=X.columns)

        # If you wish your code can have cases for different types of input and output data (discrete, real)
        # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # You may(according to your implemetation) need to call functions recursively to construct the tree. 

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int, features: pd.Series) -> Node:
        if len(y) == 0:
            return None
        if len(y.unique()) == 1:   
            return Node(value=y.iloc[0])
        if depth >= self.max_depth or len(features) == 0:
            if self.task_type == "regression":  # regression
                leaf_value = y.mean()           
            else:  # classification
                leaf_value = y.mode()[0]
            return Node(value=leaf_value)
        
        best_feature,best_threshold = opt_split_attribute(X, y, self.criterion, features)


        if best_feature is None:
            if self.task_type=="regression":
                leaf_value = y.mean()
            else:  # classification
                leaf_value = y.mode()[0]
            return Node(value=leaf_value)
        
        X_left, y_left, X_right, y_right = split_data(X, y, best_feature, best_threshold)
        
        # if split failed → make leaf
        if len(y_left) == 0 or len(y_right) == 0:
            if self.task_type == "regression":
                return Node(value=y.mean())
            else:
                return Node(value=y.mode()[0])
        
        # decide which features to pass
        if check_ifreal(X[best_feature]):
            next_features = features  # keep numeric
        else:
            next_features = features.drop(best_feature)  # drop categorical

        left_subtree = self._build_tree(X_left, y_left, depth + 1, next_features)
        right_subtree = self._build_tree(X_right, y_right, depth + 1, next_features)


        return Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)


    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """
        predictions = X.apply(lambda row: self._traverse_tree(self.root, row), axis=1)
        return predictions
    def _traverse_tree(self, node: Node, row: pd.Series):
        """
        Function to traverse the tree and return the predicted value for a given row                
        """
        if node.is_leaf():
            return node.value

        if isinstance(row[node.feature], (int, float, np.number)):
            if row[node.feature] <= node.threshold:
                return self._traverse_tree(node.left, row)
            else:
                return self._traverse_tree(node.right, row)
        else:
            if row[node.feature] == node.threshold:
                return self._traverse_tree(node.left, row)
            else:
                return self._traverse_tree(node.right, row)

        # For classification, you can return the class label or for regression, the mean of the target variable in the split.
        
        # Traverse the tree you constructed to return the predicted values for the given test inputs.

        

    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        def _plot_node(node, depth=0):
            if node.is_leaf():
                print("\t" * depth + f"Leaf: {node.value}")
                return
    
            if check_ifreal(pd.Series([node.threshold])):  # real-valued feature
                print("\t" * depth + f"?({node.feature} <= {node.threshold})")
            else:  # discrete feature
                print("\t" * depth + f"?({node.feature} == {node.threshold})")
    
            print("\t" * depth + "Yes:")
            _plot_node(node.left, depth + 1)
            print("\t" * depth + "No:")
            _plot_node(node.right, depth + 1)
        _plot_node(self.root)

        
