import numpy as np

from mlalgorithms.linear_models import LinearRegression

class Ridge(LinearRegression):
    '''
        Linear regression model with L2 regularization.
        
        Parameters
        ----------
            C: float, default=0.1
                regularization parameter
            fit_intercept: bool, default=True
                if false, intercept will not be added to the relation
        
        Attributes
        ----------
            intercept_ = float
                value of the y-intercept
            coeff_ = np.array
                coefficient vector for each feature
    '''

    def __init__(self, C=0.1, fit_intercept=True):
        super(Ridge, self).__init__(fit_intercept=fit_intercept)
        self.C = C

    def _closed_form(self, X, y):
        n_features = X.shape[1]
        # computer the dot product of X` and X
        xTx = X.T.dot(X)
        # add the regularization matrix
        M = np.eye(n_features)
        M[0, 0] = 0
        xTx = xTx + (self.C * M)
        # compute the inverse
        inv = np.linalg.pinv(xTx)
        coeff = inv.dot(X.T).dot(y)
        return coeff