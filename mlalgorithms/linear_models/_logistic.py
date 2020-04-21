import numpy as np
from math import sqrt

from mlalgorithms.linear_models import LinearRegression
from mlalgorithms.utils import fetch_batches

class LogisticRegression(LinearRegression):
    '''
        Model for classification by generating a decision boundary.

        Parameters
        ----------
            C: float, default=0.1
                regularization parameter
            penalty: string, default='l2'
                determine the type of regularization
                    values: ('l1', 'l2', 'none')
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
            cutoff: float, default=0.5
                threshold for probability of being classified in positive class.
                Must be between 0 and 1.
        
        Attributes
        ----------
            intercept_ = float
                value of the y-intercept
            coeff_ = np.array
                coefficient vector for each feature
            costs_ = np.array
                stores cost at each iteration
    '''

    def __init__(self, C=0.1, penalty='l2', learning_rate=0.01, n_iters=50, batch_size=32, fit_intercept=True, random_seed=None, cutoff=0.5):
        super(LogisticRegression, self).__init__()
        self.C = C
        self.penalty = penalty
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.batch_size = batch_size
        self._fit_intercept = fit_intercept
        self.random_seed = random_seed

        if 0 < cutoff < 1:
            self.cutoff = cutoff
        else:
            raise ValueError("Cutoff must be between 0 and 1. You provided `{}`".format(cutoff))

    def _sigmoid(self, z):
        # sigmoid activation function
        return (1 / (1 + np.exp(-z)))

    def _add_penalty(self, coeff_, bias_included=True):
        # regularization
        if bias_included:
            coeff = coeff_[1:]
        if self.penalty == 'l1':
            return self.C * np.sum(np.abs(coeff))
        elif self.penalty == 'l2':
            return self.C * np.sum(coeff ** 2) * 0.5
        return 0

    def _add_penalty_prime(self, coeff, bias_included=True):
        # gradient of regularization term
        temp = coeff.copy()
        if bias_included:
            temp[0] = 0
        if self.penalty == 'l1':
            return (self.C * np.sign(temp))
        elif self.penalty == 'l2':
            return (self.C * temp)
        return 0

    def _loss(self, X, y, coeff):
        # cross entropy loss function
        H = self._sigmoid(X.dot(coeff))
        loss = - np.mean((y * np.log(H)) + ((1 - y) * np.log(1 - H)))
        loss = loss + self._add_penalty(coeff)
        return H, loss

    def _gradient_step(self, X, y, coeff):
        H, loss = self._loss(X, y, coeff)
        err = H - y
        grad_w = X.T.dot(err)
        reg_grad = self._add_penalty_prime(coeff)
        gradients = grad_w + reg_grad
        return loss, gradients

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
            for X_batch, y_batch in fetch_batches(X, y, self.batch_size):

                J, gradients = self._gradient_step(X_batch, y_batch, coeff)
                c_.append(J)
                # update the weights
                coeff = coeff - (self.learning_rate * gradients)

            # average cost for the batch
            costs.append(np.mean(c_))

        self.intercept_ = coeff[0] if self._fit_intercept else 0
        self.coeff_ = coeff[1:]

        self.costs_ = np.array(costs)

    def predict_proba(self, X):
        # Predict the probabilty of X belonging to class 1
        z = X.dot(self.coeff_) + self.intercept_
        return self._sigmoid(z)

    def _predict(self, X):
        # predict the class of X based on the cutoff (default: 0.5)
        predicted = self.predict_proba(X)
        return np.where(predicted <= self.cutoff, 0, 1)
