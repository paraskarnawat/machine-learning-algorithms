import numpy as np
import matplotlib.pyplot as plt

from ..base import Regression


class OLSRegression(Regression):
    '''
        Multivariate Regression model using Ordinary Least Squares (OLS) method

        Parameters:
        ---------------------------------
        regularize: bool (default: True)
            if true, perform regularization
            otherwise, do not perform regularization
        C: float (default: 0.1)
            regularization parameter
    '''

    def __init__(self, regularize=True, C=0.1):
        super(OLSRegression, self).__init__(regularize=regularize, C=C)

    def _fit(self, X, y):
        # add intercept column to the feature vector
        X = self._add_intercept(X)

        _, n_features = X.shape

        self.init_parameters(n_features)

        # L is square matrix of shape (n_features + 1)
        L = np.eye(X.shape[1])
        L[0, 0] = 0

        # Compute X`.X + CL
        xTx_Li = np.dot(X.T, X) + (self.C * L)

        # Computer inv(X`.X + CL)
        inv = np.linalg.pinv(xTx_Li)

        # W = inv(X`X + CL).X`.y
        self.coeff_ = np.dot(inv.dot(X.T), y)


class GradientRegression(Regression):
    '''
        Multivariate Regression model using Gradient Descent Optimization

        Parameters:
        -----------
        n_iters: int (default: 500)
            number of iterations to tune the weights
        learning_rate: float (default: 1e-3)
            how large the steps should be taken while tuning the weights
        regularize: bool (default: True)
            if true, perform regularization
            otherwise, do not perform regularization
        C: float (default: 0.1)
            regularization parameter
        verbose: int
            level of verbosity
            0: silent
            1: show cost vs iteration plot
            2. show cost after every 100 iteration
    '''

    def __init__(self, n_iters=500, learning_rate=1e-3, regularize=True, C=0.1, verbose=0):
        super(GradientRegression, self).__init__(regularize=regularize, C=C)
        self.n_iters = n_iters
        self.eta = learning_rate
        self.verbose = verbose

    def __cost(self, y_true, y_pred):
        pred_cost = 0.5 * np.mean(np.power(y_true - y_pred, 2))
        reg_cost = self.C * 0.5 * np.sum(np.power(self.coeff_[1:], 2))

        return (pred_cost + reg_cost)

    def __plot_history(self):
        _, ax = plt.subplots(1, 1, sharey=True)

        ax.plot(range(self.n_iters), self.costs)
        ax.set_title("Cost through iterations")
        ax.legend(f'Learning Rate: {self.eta}', loc='best')
        ax.set_xlabel("Number of Iterations")
        ax.set_ylabel("Cost")

        plt.show()

    def _fit(self, X, y):
        # add intercept term to the feature vector
        X = self._add_intercept(X)
        n_samples, n_features = X.shape
        costs = []

        self.init_parameters(n_features)

        # perform batch gradient descent
        for i in range(self.n_iters):

            # hypothesis
            y_pred = X.dot(self.coeff_)

            # save the cost history
            costs.append(self.__cost(y, y_pred))
            # derivate of cost function with respect to weights
            grad_coeff_cost = X.T.dot(y_pred - y)

            # derivate of regularization term with respect to weights
            grad_coeff_reg = self.C * self.coeff_

            # the bias weight should not be regularized, or updated by regularization
            grad_coeff_reg[0] = 0

            grad = grad_coeff_cost + grad_coeff_reg

            # update the weights
            self.coeff_ = self.coeff_ - (self.eta / n_samples) * (grad)

            if (i + 1) % 100 == 0 and self.verbose == 2:
                print(f"Iteration: {i + 1:4}, cost: {costs[-1]}")

        self.costs = costs
        if 0 < self.verbose < 2:
            self.__plot_history()