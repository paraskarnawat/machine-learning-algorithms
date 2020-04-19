import numpy as np

from mlalgorithms.base import Model
from mlalgorithms.utils import fetch_batches, polynomial_features, train_test_split
from mlalgorithms.metrics import mean_squared_error, binary_cross_entropy

class BaseRegression(Model):

    def __init__(self, eta=0.001, penalty=None, C=0.01, tolerance=1e-5, n_iters=1000, degree=None, mini_batches=64):
        self.C = C
        self.eta = eta
        self.penalty = penalty
        self.tolerance = tolerance
        self.n_iters = n_iters
        self.degree = degree
        self.mini_batches = mini_batches

        self.weights_ = None
        self._cost_function = mean_squared_error

    def _init_cost(self):
        raise NotImplementedError()

    def _loss(self, X, y):
        raise NotImplementedError()

    def _add_penalty(self):
        if self.penalty == 'L1':
            return self.C * np.sum(np.abs(self.weights_[1:]))
        elif self.penalty == 'L2':
            return self.C * 0.5 * np.sum(np.power(self.weights_[1:], 2))
        return 0

    def _grad_penalty(self):
        temp_weights = self.weights_.copy()
        temp_weights[0] = 0
        if self.penalty == 'L1':
            return self.C * (temp_weights / np.linalg.norm(temp_weights))
        elif self.penalty == 'L2':
            return self.C * (temp_weights)
        return 0

    def _add_intercept(self, X):
        ones = np.ones((X.shape[0], 1))
        return np.hstack((ones, X))

    def _fit(self):
        self._init_cost()

        if self.degree is not None:
            self.X = polynomial_features(self.X, degree=self.degree)

        self.X = self._add_intercept(self.X)
        
        n_samples, n_features = self.X.shape

        limit = 1. / n_features

        self.weights_ = np.random.uniform(- limit, limit, n_features)

        return self._gradient_descent(n_samples)

    def _gradient_descent(self, n_samples):
        
        train_, validation_ = [], []

        for i in range(self.n_iters):
            t, v = [], []
            prev_weights = self.weights_.copy()
            for X_batch, y_batch in fetch_batches(self.X, self.y, batch_size=self.mini_batches):

                X_train, X_validation, y_train, y_validation = train_test_split(X_batch, y_batch, test_size=0.3)

                err, training_cost = self._loss(X_train, y_train)

                t.append(training_cost)

                _, validation_cost = self._loss(X_validation, y_validation)
                
                v.append(validation_cost)

                gradients = X_train.T.dot(err)

                gradients += self._grad_penalty()
                
                self.weights_ -= (self.eta * gradients)

            train_.append(np.mean(t))
            validation_.append(np.mean(v))

            if (i + 1) % 100 == 0:
                print(f"iteration: {i + 1}, loss: {training_cost}, validation-loss: {validation_cost}")

            if self.tolerance and np.allclose(prev_weights, self.weights_, atol=self.tolerance):
                break
            
        return train_, validation_

    def _predict(self, X):
        if self.degree is not None:
            X = polynomial_features(X, degree=self.degree)
        X = self._add_intercept(X)

        return X.dot(self.weights_)

class LinearRegression(BaseRegression):

    def _init_cost(self):
        self._cost_function = mean_squared_error
    
    def _loss(self, X, y):
        y_pred = X.dot(self.weights_)
        err = y_pred - y
        loss = self._cost_function(y, y_pred)
        loss = loss + self._add_penalty()
        return err, loss

class LogisticRegression(BaseRegression):

    def _init_cost(self):
        self._cost_function = binary_cross_entropy

    def _loss(self, X, y):
        y_pred = self.sigmoid(X.dot(self.weights_))
        loss = self._cost_function(y, y_pred)
        err = y_pred - y
        loss = loss + self._add_penalty()
        return err, loss

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        if self.degree is not None:
            X = polynomial_features(X, degree=self.degree)
        X = self._add_intercept(X)

        return X.dot(self.weights_)

    def _predict(self, X):
        if self.degree is not None:
            X = polynomial_features(X, degree=self.degree)
        X = self._add_intercept(X)

        h = self.sigmoid(X.dot(self.weights_))
        return h