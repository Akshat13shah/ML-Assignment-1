"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    return pd.get_dummies(X)
    

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    if pd.api.types.is_float_dtype(y):
        # Floats are continuous
        return True
    elif pd.api.types.is_integer_dtype(y):
        # Integers with only a few unique values → categorical
        return y.nunique() > 10
    return False
    


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    counts = Y.value_counts().values
    if counts.sum() > 0:
        probabilities = counts / counts.sum()
    else:
        probabilities = 0 
    return -np.sum(probabilities * np.log2(probabilities + 1e-9)) # here we used a small value to avoid log(0)
    


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    counts = Y.value_counts().values
    if counts.sum() > 0:
        probabilities = counts / counts.sum()
    else:
        probabilities = 0 
    return 1 - np.sum(probabilities ** 2)
    

def mse(Y: pd.Series) -> float:
    """
    Function to calculate the mean squared error
    """    
    return np.mean((Y - Y.mean()) ** 2)


def information_gain(y, y_left, y_right, criterion):
    if criterion == "entropy":
        impurity_func = entropy
    elif criterion == "gini":
        impurity_func = gini_index
    elif criterion == "mse":
        impurity_func = mse
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    parent = impurity_func(y)
    left_imp = impurity_func(y_left)
    right_imp = impurity_func(y_right)

    # weighted average impurity
    weighted_child_imp = (len(y_left)/len(y)) * left_imp + (len(y_right)/len(y)) * right_imp

    return parent - weighted_child_imp


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split upon.
    Handles both continuous (real) and discrete features.
    
    Returns:
    - best_feature: feature name
    - best_threshold: threshold (for continuous) or category (for discrete)
    """

    best_gain = -float("inf")
    best_feature = None
    best_threshold = None

    for feature in features:
        column = X[feature]

        #  Continuous feature
        if check_ifreal(column):
            values = np.sort(column.unique())
            thresholds = (values[:-1] + values[1:]) / 2  # midpoints

            for t in thresholds:
                mask = column <= t
                if mask.sum() == 0 or (~mask).sum() == 0:
                    continue

                #  Split y into left and right
                y_left, y_right = y[mask], y[~mask]

                gain = information_gain(y,y_left, y_right, criterion)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = t

        # Discrete feature
        else:
            for v in np.unique(column):
                mask = column == v
                if mask.sum() == 0 or (~mask).sum() == 0:
                    continue

                y_left, y_right = y[mask], y[~mask]

                gain = information_gain(y,y_left, y_right, criterion)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = v

    return best_feature, best_threshold



def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.
    if check_ifreal(X[attribute]):
        left_mask = X[attribute] <= value
        right_mask = X[attribute] > value
    else:
        left_mask = X[attribute] == value
        right_mask = X[attribute] != value

    return (
        X[left_mask], y[left_mask],
        X[right_mask], y[right_mask]
    )

