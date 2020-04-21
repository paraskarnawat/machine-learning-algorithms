import numpy as np

from mlalgorithms._base import Model


class LinearRegression(Model):
    '''
        Estimate parameters to model the relationship between features and target.

        Parameters
        ----------
            fit_intercept: bool, default=True
                if false, intercept will not be added to the relation

        Attributes
        ----------
            intercept_ = float
                value of the y-intercept
            coeff_ = np.array
                coefficient vector for each feature
    '''

    def __init__(self, fit_intercept=True):
        super(LinearRegression, self).__init__()
        self._fit_intercept = fit_intercept

    def _add_intercept(self, X):
        n_samples = X.shape[0]
        shape = (n_samples, 1)
        _intercept = np.ones(shape) if self._fit_intercept else np.zeros(shape)
        return np.hstack((_intercept, X))

    def _closed_form(self, X, y):
        # dot product of X` and X
        xTx = X.T.dot(X)
        # compute the inverse of X`.X
        inv = np.linalg.pinv(xTx)
        # coeff = inv(X`X).X`.y
        coeff = inv.dot(X.T).dot(y)
        return coeff

    def _fit(self, X, y):
        X = self._add_intercept(X)
        coeff = self._closed_form(X, y)
        self.intercept_ = coeff[0:] if self._fit_intercept else 0
        self.coeff_ = coeff[1:]

    def _predict(self, X):
        return (X.dot(self.coeff_) + self.intercept_)
