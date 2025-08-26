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
    return pd.api.types.is_numeric_dtype(y)
    


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

def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion 
    (entropy, gini index or MSE)
    """
    # parent impurity
    if criterion == "information_gain":
        parent_impurity = entropy(Y)
    elif criterion == "gini_index":
        parent_impurity = gini_index(Y)
    elif criterion == "MSE":
        parent_impurity = mse(Y)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    # splitting
    values, counts = np.unique(attr, return_counts=True)
    #print(values, counts)
    weighted_impurity = 0
    for v, count in zip(values, counts):
        subset_Y = Y[attr == v]
        #print(subset_Y)
        weight = count / len(Y)
        if criterion == "information_gain":
            weighted_impurity += weight * entropy(subset_Y)
        elif criterion == "gini_index":
            weighted_impurity += weight * gini_index(subset_Y)
        elif criterion == "MSE":
            weighted_impurity += weight * mse(subset_Y)

    return parent_impurity - weighted_impurity


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """
    best_gain = -float("inf")
    best_feature = None
    best_threshold = None

    for feature in features:
        column = X[feature]

        if check_ifreal(column):  # continuous feature
            c_srt = column.sort_values()
            thresholds = [(c_srt.iloc[i]+c_srt.iloc[i+1])/2 for i in range (len(c_srt)-1)]
            for t in thresholds:
                left_mask = column <= t
                right_mask = column > t
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                gain = information_gain(y, column <= t, criterion)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = t
        else:  # discrete feature
            values = np.unique(column)
            for v in values:
                gain = information_gain(y, column ==  v , criterion)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = v

    return best_feature, best_threshold
    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).

   


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

