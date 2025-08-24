from typing import Union
import pandas as pd
import numpy as np

def conf_mat(act,tr, fl):
    lbl = []
    for i in act:
        for j in act:
            lbl.append([i,j])
    c_m = pd.DataFrame(0, index = act, columns = act)
        
    for i in lbl:
        for k,j in zip(tr,fl):
                if i == [k,j]:
                    c_m.loc[k,j] += 1
    return c_m


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    # TODO: Write here
    act = np.unique(y)
    c_m = conf_mat(act, y, y_hat)
    a = 0
    t = 0
    for i in c_m.index:
        for j in c_m.columns:
            if i == j:
                a += c_m.loc[i,j]
            t += c_m.loc[i,j]
    pass
    return (a/t)


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size
    act = np.unique(y)
    c_m = conf_mat(act, y, y_hat)
    p = []
    for j in c_m.columns:
        s_d = 0
        s = 0
        for i in c_m.index:
            if i == j:
                s_d += c_m.loc[i,j]
            s += c_m.loc[i,j]
        p.append(s_d/s)
    p_ = np.mean(p)
    pass
    return p_


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size
    act = np.unique(y)
    c_m = conf_mat(act, y, y_hat)
    r = []
    for i in c_m.index:
        s_d = 0
        s = 0
        for j in c_m.columns:
            if i == j:
                s_d += c_m.loc[i,j]
            s += c_m.loc[i,j]
        r.append(s_d/s)
    r_ = np.mean(r)
    pass
    return r_


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.size == y.size
    s = 0
    for i in range (len(y)):
        s += (y[i] - y_hat[i])**2
    s_m = (1/len(y))*s
    s_rm = np.sqrt(s_m)
    pass
    return s_rm


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size
    s = 0
    for i in range (len(y)):
        s += abs(y[i] - y_hat[i])
    s_m = (1/len(y))*s
    s_rm = np.sqrt(s_m)
    pass
    return s_rm
