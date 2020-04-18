import numpy as np
import matplotlib.pyplot as plt

from ..base import Classification

class BinaryLogistic(Classification):
    '''
        Binary classification model using Gradient Descent Optimization

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
        super(BinaryLogistic, self).__init__(regularize=regularize, C=C)
        self.n_iters = n_iters
        self.eta = learning_rate
        self.verbose = verbose

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
            y_pred = self.sigmoid(y_pred)

            cost = - y.dot(np.log(y_pred)) - (1 - y).dot(np.log(1 - y_pred))

            cost = cost + np.sum(np.power(self.coeff_[1:], 2))

            # save the cost history
            costs.append(cost)
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