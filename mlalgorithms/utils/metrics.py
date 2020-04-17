import numpy as np

def mean_squared_error(y_true, y_pred):
    # Compute the mean squared error between actual and predicted value
    return np.mean(np.power(y_true - y_pred, 2))

def r2_score(y_true, y_pred):
    # Compute the model performance: 1 - (SSR / SST)
    # SSR : Residual Sum of Squares
    # SST : Total Sum of Squares
    residual_sos = np.sum(np.power(y_true - y_pred, 2))
    total_sos = np.sum(np.power(y_true - np.mean(y_true), 2))
    return (1 - (residual_sos / total_sos))