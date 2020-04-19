import numpy as np

class Model:

    X = None
    y = None

    fitted = True

    def __init__(self):
        pass

    def _feed(self, X, y=None):
        self.X = X
        if X.ndim == 1:
            self.X = X[:, np.newaxis]
        self.y = y

    def fit(self, X, y=None):
        self._feed(X, y)
        t, v = self._fit()
        self.fitted = True
        return t, v

    def _fit(self):
        raise NotImplementedError()

    def predict(self, X):
        if not self.fitted:
            raise AttributeError("You must fit the model first.")
        return self._predict(X)

    def _predict(self, X):
        raise NotImplementedError()
