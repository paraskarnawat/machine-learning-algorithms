import numpy as np
from math import sqrt

from mlalgorithms.linear_models import LinearRegression
from mlalgorithms.utils import fetch_batches
from mlalgorithms.metrics import binary_cross_entropy


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

    def __init__(self, C=0.1, penalty='l2', learning_rate=0.01, n_iters=50, batch_size=32, fit_intercept=True, cutoff=0.5):
        super(LogisticRegression, self).__init__()
        self.C = C
        self.penalty = penalty
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.batch_size = batch_size
        self._fit_intercept = fit_intercept

        if 0 < cutoff < 1:
            self.cutoff = cutoff
        else:
            raise ValueError(
                "Cutoff must be between 0 and 1. You provided `{}`".format(cutoff))

    def _sigmoid(self, z):
        # sigmoid activation function
        return (1 / (1 + np.exp(-z)))

    def _add_penalty(self, coeff_, bias_included=True):
        # regularization
        if bias_included:
            coeff = coeff_[1:]
        else:
            coeff = coeff_
        if self.penalty == 'l1':
            if bias_included:
                return self.C * np.sum(np.abs(coeff))
            else:
                return self.C * np.sum(np.sum(np.abs(coeff)))
        elif self.penalty == 'l2':
            if bias_included:
                return self.C * np.sum(coeff ** 2) * 0.5
            else:
                return self.C * np.sum(np.sum(coeff ** 2)) * 0.5
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
        loss = binary_cross_entropy(y, H)
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

        # fetch the number of classes
        self.n_classes = len(np.unique(y))

        # if multi-class classifcation, train one vs all
        if self.n_classes > 2:
            coeff = np.zeros((self.n_classes, n_features))
            costs = []
            intercepts = []
            # training for each class
            for c in range(self.n_classes):
                y_ = (y == c).astype(int)
                coeff[c, ], cost_ = self._train(X, y_, coeff[c, ].T)
                intercepts.append(coeff[c, 0])
                costs.append(cost_)

            self.intercept_ = np.array(intercepts)
            self.coeff_ = coeff[:, 1:].T
        # binary classification
        else:
            coeff = np.zeros((n_features, ))
            coeff, costs = self._train(X, y, coeff)
            self.intercept_ = 0 if not self._fit_intercept else coeff[0]
            self.coeff_ = coeff[1:]
        
        self.costs_ = np.array(costs)

    def _train(self, X, y, coeff):
        _, n_features = X.shape

        # save the costs per iteration
        costs = []

        # initialize the coefficients
        coeff = np.zeros((n_features, ))

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
        return coeff, costs

    def predict_proba(self, X):
        # Predict the probabilty of X belonging to class 1
        z = X.dot(self.coeff_) + self.intercept_
        return self._sigmoid(z)

    def _predict(self, X):
        # predict the class of X based on the cutoff (default: 0.5)
        predicted = self.predict_proba(X)
        if self.n_classes > 2:
            return np.argmax(predicted, axis=1)
        return np.where(predicted <= self.cutoff, 0, 1)
