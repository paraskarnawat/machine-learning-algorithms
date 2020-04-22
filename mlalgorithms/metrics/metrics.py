import numpy as np

EPS = 1e-15

def mean_squared_error(y_true, y_pred):
    return np.mean(np.power(y_pred - y_true, 2))

def binary_cross_entropy(y_true, y_pred):
    # Avoid log(0) error
    y_pred = np.clip(y_pred, EPS, 1 - EPS)
    return -np.mean((y_true * np.log(y_pred)) + ((1 - y_true) * np.log(y_pred)))

def categorical_cross_entropy(y_true, y_pred):
    # Avoid log(0) error
    y_pred = np.clip(y_pred, EPS, 1 - EPS)
    return - np.mean(y_true * np.log(y_pred))