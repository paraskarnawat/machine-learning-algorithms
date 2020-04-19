import numpy as np

EPS = 1e-15


def accuracy_score(y_true, y_pred):
    '''
        Accuracy of the classification model.

        Parameters
        ----------
            y_true: np.array
                ground truth values
            y_pred: np.array
                predicted values
    '''
    acc = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return acc


def binary_cross_entropy(y_true, y_pred):
    '''
        Binary Cross Entropy as cost function for binary classification.

        Parameters
        ----------
            y_true: np.array
                ground truth values
            y_pred: np.array
                predicted values
    '''
    y_pred = np.clip(y_pred, EPS, 1 - EPS)
    return np.mean(- np.sum((y_true * np.log(y_pred)) + ((1 - y_true) * np.log(1 - y_pred))))


def mean_squared_error(y_true, y_pred):
    '''
        Mean squared error for regression model.

        Parameters
        ----------
            y_true: np.array
                actual values
            y_pred:
                predicted values
    '''
    return 0.5 * np.mean(np.power(y_true - y_pred, 2))


def r2_score(y_true, y_pred):
    '''
        Regression model score.
            Statistical measure of how close the data are to the fitted regression line.

        Parameters
        ----------
            y_true: np.array
                actual values
            y_pred:
                predicted values
    '''
    y_bar = np.mean(y_true)
    ssr = np.sum(np.power(y_true - y_pred, 2))
    sst = np.sum(np.power(y_true - y_bar, 2))
    return (1 - (ssr / sst))


def confusion_matrix(y_true, y_pred):
    '''
        Summary of prediction results on classification problem.
        Allows easy identification of confusion between classes.

        Example (for binary classification)

            |   Actual  |
            |  0  |  1  |
        -----------------
          0 |     |     |
        ----------------- Predicted
          1 |     |     |
        -----------------

        Parameters
        ----------
            y_true: np.array
                actual values
            y_pred:
                predicted values
    '''
    K = len(np.unique(y_true))
    cfmat = np.zeros((K, K))
    for true, pred in zip(y_true, y_pred):
        cfmat[true, pred] += 1
    return cfmat
