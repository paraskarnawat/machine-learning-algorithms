import numpy as np
from ..utils.metrics import mean_squared_error


class Ridge:
    '''
        Regularization for Ridge regression

        Parameters:
        -----------------------------------
        lambda_ (regularization parameter) : float
            Larger values cause underfitting, smaller values cause overfitting. 

    '''

    def __init__(self, lambda_=0.01):
        self.lambda_ = lambda_

    def __call__(self, W, n_samples):
        return (self.lambda_ * (np.sum(np.power(W[1:], 2)))) / (2 * n_samples)

    def gradient(self, W):
        return self.lambda_ * W


class Regression(object):
    '''
        Base Class for Linear Regression:
            Predicting dependent variable `y` on independent feature(s) `X`.

        Parameters:
        ---------------------------------
        learning_rate: float
            how farther the weights will be updated
        n_iterations: int
            how long the algorithm will tune the weights
    '''

    def __init__(self, learning_rate=0.01, n_iterations=5000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.cost_history = None
        self._is_fitted = False
        self.regularization = lambda x, y: 0
        self.regularization.gradient = lambda x: 0

    def _initialize_weights(self, n_features):
        # Initializing the weight vector to zero (including the bias term)
        self.weights = np.zeros((n_features, ))

    def _add_ones(self, X):
        # Add 1s column to the input features
        return np.insert(X, 0, 1, axis=1)

    def _update_weight(self, grad_W, n_samples):
        # Simultaneous update of weights using gradients of the weights
        self.weights -= (((self.learning_rate / n_samples) *
                          grad_W) + self.regularization.gradient(self.weights))

    def fit(self, X, y):
        # Fit the model to X and y
        X = self._add_ones(X)
        n_samples, n_features = X.shape
        costs = []

        self._initialize_weights(n_features)

        for iteration in range(self.n_iterations):

            y_hat = X.dot(self.weights)
            cost = mean_squared_error(y, y_hat) / 2.
            cost = cost + self.regularization(self.weights, n_samples)
            costs.append(cost)

            grad_w = X.T.dot(y_hat - y)

            self._update_weight(grad_w, n_samples)

            if (iteration + 1) % 100 == 0:
                print(f"[INFO] iteration: {iteration + 1:04}, cost: {cost}")

        self.cost_history = np.array(costs)
        self._is_fitted = True

    def predict(self, X):
        # Predict the target value for independent features, X
        if not self._is_fitted:
            print("[ERROR] Train the model first using `fit` method.")
        else:
            X = self._add_ones(X)
            return X.dot(self.weights)


class LinearRegression(Regression):
    '''
        Simple Multivariate Regression

        Parameters:
        ---------------------------------
        learning_rate: float
            how farther the weights will be updated
        n_iterations: int
            how long the algorithm will tune the weights
        direct: bool
            if true, use the normal equation method for optimization
            otherwise, use the batch gradient descent algorithm
    '''

    def __init__(self, learning_rate=0.01, n_iterations=5000, direct=False):
        super(LinearRegression, self).__init__(
            learning_rate=learning_rate, n_iterations=n_iterations)
        self.direct = direct

    def fit(self, X, y):

        if not self.direct:
            super(LinearRegression, self).fit(X, y)
        else:
            # Normal equation, W = (inv(X.X`)).X`.y
            # Used pinv() to avoid the case if the matrix X.X` in singular
            X = self._add_ones(X)
            self.weights = (((np.linalg.pinv(X.T.dot(X))).X.T).dot(y))
            self._is_fitted = True


class RidgeRegression(Regression):
    '''
        Regularized Multivariate Regression

        Parameters:
        ---------------------------------
        learning_rate: float
            how farther the weights will be updated
        n_iterations: int
            how long the algorithm will tune the weights
        lambda_ : float
            regularization parameter
    '''

    def __init__(self, learning_rate=0.01, n_iterations=5000, lambda_=0.01):
        super(RidgeRegression, self).__init__(
            learning_rate=learning_rate, n_iterations=n_iterations)
        self.lambda_ = lambda_
        self.regularization = Ridge(lambda_=lambda_)
