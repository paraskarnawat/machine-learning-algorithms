import numpy as np
import cvxopt

from mlalgorithms._base import Model
from ._kernels import *

cvxopt.solvers.options['show_progress'] = False

class SVC(Model):
    '''
        Support Vector Classifier for binary classification

        Parameters
        ----------
            C: float, default=1
                regularization parameter;
                High value of C implies increase in cost of misclassification, i.e. smaller margins
            kernel: string, default='linear'
                specify the kernel used to transform the data
                Allowed values:
                    1. 'linear': for linear kernel
                    2. 'poly': for polynomial kernel
                    3. 'rbf' for radial basic function kernel
            degree: int, default=2
                specify the degree for the polynomial kernel, will be ignored by other kernel
            gamma: float, default=0.01
                used in rbf kernel, will be ignored by other kernels
            const: float, default=4
                used as constant for the polynomial kernel
    '''

    def __init__(self, C=1, kernel='linear', degree=2, gamma=0.01, const=4):
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.const = const

        supported_kernels = {
            'linear': linear_kernel,
            'poly': polynomial_kernel,
            'rbf': rbf_kernel
        }

        if kernel not in supported_kernels.keys():
            raise ValueError("`{}` is unknown kernel; supported kernels are: `{}`".format(
                kernel, supported_kernels.keys()))

        self.kernel = supported_kernels[kernel](
            gamma=self.gamma, const=self.const, power=self.degree)

        self.lagrange_multipliers = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.intercept_ = None

    def _fit(self, X, y):

        n_samples, n_features = X.shape

        # Change the Class 0 to negative class
        y[y == 0] = -1

        # Initialize the kernel matrix
        kernel_matrix = np.zeros((n_samples, n_samples))

        # Fill the kernel matrix
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = self.kernel(X[i], X[j])

        # Initialize the minimization function and constraints
        P = cvxopt.matrix(np.outer(y, y) * kernel_matrix, tc='d')
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples), tc='d')
        b = cvxopt.matrix(0, tc='d')

        if not self.C:
            G = cvxopt.matrix(np.identity(n_samples) * -1)
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            G_max = np.identity(n_samples) * -1
            G_min = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((G_max, G_min)))
            h_max = np.zeros((n_samples, 1))
            h_min = np.ones((n_samples, 1)) * self.C
            h = cvxopt.matrix(np.vstack((h_max, h_min)))

        # Solve the constrained minimization problem
        solved = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Fetch the lagrange multipliers
        lagr_mult = np.ravel(solved['x'])

        # Extract support vectors
        idx = lagr_mult > 1e-7

        # Get the lagrange multipliers, labels and features for the support vectors
        self.lagrange_multipliers = lagr_mult[idx]
        self.support_vector_labels = y[idx]
        self.support_vectors = X[idx]

        # Calculate the intercept term
        self.intercept_ = self.support_vector_labels[0]

        for i in range(len(self.lagrange_multipliers)):
            self.intercept_ -= self.lagrange_multipliers[i] * self.support_vector_labels[i] * self.kernel(
                self.support_vectors[i], self.support_vectors[0])

    def _predict(self, X):
        y_pred = []

        for sample in X:
            prediction = 0
            for i in range(len(self.lagrange_multipliers)):
                prediction += self.lagrange_multipliers[i] * self.support_vector_labels[i] * self.kernel(
                    self.support_vectors[i], sample)
            prediction += self.intercept_
            y_pred.append(np.sign(prediction))

        y_pred = np.array(y_pred)
        y_pred[y_pred == -1] = 0
        return y_pred
