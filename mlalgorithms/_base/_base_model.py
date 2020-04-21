import numpy as np
from abc import abstractmethod

class Model(object):
    '''
        Base class for the Model object.

        Attributes
        ----------
            _fitted: bool
                check whether the model is trained, or not.
    '''
    def __init__(self):
        self._fitted = False

    def _check(self, X, y=None):
        # check if X is 2 dimensional or not
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # sanity check for number of instances in X and y
        if y is not None and (y.shape[0] != X.shape[0]):
            raise ValueError("`X` and `y` must have same number of instances")
        
        return X, y

    def fit(self, X, y=None):
        # fit the model to X and y
        X, y = self._check(X, y)
        self._fit(X, y)
        self._fitted = True

    def predict(self, X):
        # predict values for unknown X
        self._check(X)
        return self._predict(X)

    @abstractmethod
    def _fit(self, X, y):
        # to be implemented by the subclass
        pass

    @abstractmethod
    def _predict(self, X):
        # to be inherited by the subclass
        pass