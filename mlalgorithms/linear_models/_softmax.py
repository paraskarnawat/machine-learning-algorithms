import numpy as np
from math import sqrt

from mlalgorithms.linear_models import LogisticRegression
from mlalgorithms.utils import one_hot_encoding, unhot_encoding, fetch_batches
from mlalgorithms.metrics import categorical_cross_entropy


class SoftmaxRegression(LogisticRegression):
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

        Attributes
        ----------
            intercept_ = float
                value of the y-intercept
            coeff_ = np.array
                coefficient vector for each feature
            costs_ = np.array
                stores cost at each iteration
    '''

    def __init__(self, C=0.1, penalty='l2', learning_rate=0.01, n_iters=50, batch_size=32, fit_intercept=True, random_seed=None):
        super(SoftmaxRegression, self).__init__()
        self.C = C
        self.penalty = penalty
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.batch_size = batch_size
        self._fit_intercept = fit_intercept
        self.random_seed = random_seed

    def _softmax(self, z):
        # softmax activation function
        e_x = np.exp(z - np.max(z, axis=1, keepdims=True))
        out = e_x / np.sum(e_x, axis=1, keepdims=True)
        return out

    def _loss(self, X, y, coeff, intercept_):
        # cross entropy loss function
        H = self._softmax(X.dot(coeff) + intercept_)
        loss = categorical_cross_entropy(y, H)
        loss = loss + self._add_penalty(coeff, bias_included=False)
        return H, loss

    def _gradient_step(self, X, y, coeff, intercept_):
        H, loss = self._loss(X, y, coeff, intercept_)
        err = H - y
        grad_w = X.T.dot(err)
        reg_grad = self._add_penalty_prime(coeff, bias_included=False)
        gradients = grad_w + reg_grad
        return err, loss, gradients

    def _fit(self, X, y):
        X = self._add_intercept(X)
        _, n_features = X.shape

        # save the costs per iteration
        costs = []

        y_enc, n_classes = one_hot_encoding(y)

        # initialize the coefficients
        limit = 1. / sqrt(n_features)
        np.random.seed(seed=self.random_seed)
        coeff = np.random.uniform(-limit, limit, (n_features, n_classes))
        intercept = np.random.random_sample() if self._fit_intercept else 0
        # perform gradient descent for `n_iter` iterations
        for _ in range(self.n_iters):
            # save the cost for each batch
            c_ = []
            for X_batch, y_batch in fetch_batches(X, y_enc, self.batch_size):

                err, J, gradients = self._gradient_step(
                    X_batch, y_batch, coeff, intercept)
                c_.append(J)
                # update the weights
                coeff = coeff - (self.learning_rate * gradients)
                intercept = intercept - \
                    (self.learning_rate * np.sum(err, axis=0))
            # average cost for the batch
            costs.append(np.mean(c_))

        self.intercept_ = intercept if self._fit_intercept else 0
        self.coeff_ = coeff

        self.costs_ = np.array(costs)

    def predict_proba(self, X):
        # Predict the probabilty of X belonging to class 1
        z = X.dot(self.coeff_) + self.intercept_
        return self._sigmoid(z)

    def _predict(self, X):
        # predict the class of X based on the cutoff (default: 0.5)
        predicted = self.predict_proba(X)
        return unhot_encoding(predicted)
