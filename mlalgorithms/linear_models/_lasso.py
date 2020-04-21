import numpy as np
from math import sqrt

from mlalgorithms.linear_models import LinearRegression
from mlalgorithms.utils import fetch_batches


class Lasso(LinearRegression):
    '''
        Linear Regression model with L1 regularization.
        Lasso uses Gradient Descent as optimization algorithm.

        Parameters
        ----------
            C: float, default=0.1
                regularization parameter
            learning_rate: float, default=0.001
                learning_rate defines how large the steps should be
                taken to tune the coefficients and intercept
            n_iters: int, default=50
                number of iterations to tune the weights
            batch_size: int, default=32
                number of instances to work on during each iteration
                    if = 1, stochastic gradient descent is performed.
                    if = n_samples, batch gradient descent is performed.
                    otherwise, mini-batch gradient descent is performed.
            fit_intercept: bool, default=True
                if false, intercept will not be added to the relation
            random_seed: int, default=None
                seed for random values

        Attributes
        ----------
            intercept_ = float
                value of the y-intercept
            coeff_ = np.array
                coefficient vector for each feature
            costs_ = np.array
                stores cost at each iteration
    '''

    def __init__(self, C=0.1, learning_rate=0.01, n_iters=50, batch_size=32, fit_intercept=True, random_seed=None):
        super(Lasso, self).__init__(fit_intercept=fit_intercept)
        self.C = C
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.random_seed = random_seed

    def _gradient_step(self, X, y, coeff):
        # Compute the hypothesis
        H = X.dot(coeff)
        # Copy the weights
        temp_coeff = coeff.copy()
        # Compute the error
        err = H - y
        # Mean Squared Error + Regularization Cost
        J = np.mean(err ** 2) + (self.C * np.sum(np.abs(coeff)))
        # Compute gradients with respect to predictions
        grads = X.T.dot(err)
        # Compute gradients with respect to regularization
        reg_grads = np.sign(temp_coeff)
        # Compute the updates for the weights
        updates = grads + (self.C * reg_grads)
        return J, updates

    def _fit(self, X, y):
        X = self._add_intercept(X)
        _, n_features = X.shape

        # save the costs per iteration
        costs = []

        # initialize the coefficients
        limit = 1. / sqrt(n_features)
        np.random.seed(seed=self.random_seed)
        coeff = np.random.uniform(-limit, limit, (n_features, ))

        # perform gradient descent for `n_iter` iterations
        for _ in range(self.n_iters):
            # save the cost for each batch
            c_ = []
            for X_batch, y_batch in fetch_batches(X, y, batch_size=self.batch_size):

                J, gradients = self._gradient_step(X_batch, y_batch, coeff)
                c_.append(J)
                # update the weights
                coeff = coeff - (self.learning_rate * gradients)

            # average cost for the batch
            costs.append(np.mean(c_))

        self.intercept_ = coeff[0] if self._fit_intercept else 0
        self.coeff_ = coeff[1:]

        self.costs_ = np.array(costs)

    def _predict(self, X):
        return ((X.dot(self.coeff_)) + self.intercept_)

