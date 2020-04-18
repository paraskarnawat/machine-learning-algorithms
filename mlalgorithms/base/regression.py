import numpy as np
from abc import abstractmethod


class Regression(object):
    '''
        Base class for Regression models.

        Attributes:
        -----------
            coeff_ : array (float)
                the coefficient (weights) vector

            __is_fitted : bool
                to check if the model is fitted or not

            __summary: dict
                'mse': float
                    mean squared error on training set
                'r2': float
                    r-squared; meausre of how well the model is fitting on training data
                'coefficients': list of tuples
                    weight for each feature
                'intercept': float
                    the intercept/ bias term
        Parameters:
        -----------
            regularize: bool (default: True)
                if true, use regularization
            C: float (default: 0.01)
                value of regularization parameter
    '''

    def __init__(self, regularize=False, C=0.01):
        self.__is_fitted = False
        self.__summary = dict()
        self.regularize = regularize
        self.C = C if regularize else 0

    def _add_intercept(self, X):
        return np.insert(X, 0, 1, axis=1)

    def init_parameters(self, n_features):
        limit = 1. / n_features
        self.coeff_ = np.random.uniform(-limit, limit, size=(n_features, ))

    def fit(self, X, y):

        X_b = X

        self._fit(X, y)
        self.__is_fitted = True

        n_features = X_b.shape[1]

        self.__summary['mse'] = self.mean_squared_error(X_b, y)
        self.__summary['r2-score'] = self.r2_score(X_b, y)
        self.__summary['coefficients'] = [(f'Feature {i + 1}', self.coeff_[1 + i]) for i in range(n_features)]
        self.__summary['intercept'] = self.coeff_[0]
        self.summary()

    def predict(self, X):
        if not self.__is_fitted:
            raise AttributeError(
                "Model is not trained yet. Train the model first!")
        y_pred = self._predict(X)
        return y_pred

    @abstractmethod
    def _fit(self, X, y):
        pass

    def _predict(self, X):
        # add the intercept column to the feature vector
        X = self._add_intercept(X)

        # compute X.W
        return X.dot(self.coeff_)

    def summary(self):
        print(f"Mean Squared Error: {self.__summary['mse']}")
        print(f"R2 Score: {self.__summary['r2-score']}")
        print("Coefficients: ")
        for s, c in self.__summary['coefficients']:
            print(f"   {s}: {c}")
        print(f"Intercept: {self.__summary['intercept']}")

    def mean_squared_error(self, X, y):
        y_pred = self.predict(X)
        err = y - y_pred
        return np.mean(np.power(err, 2))

    def r2_score(self, X, y):
        y_pred = self.predict(X)
        err = y - y_pred
        y_bar = np.mean(y)
        sst = np.sum(np.power(y - y_bar, 2))
        ssr = np.sum(np.power(err, 2))

        return (1 - ssr / sst)