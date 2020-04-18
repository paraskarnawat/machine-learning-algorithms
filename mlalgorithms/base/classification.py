import numpy as np
from abc import abstractmethod


class Classification(object):
    '''
        Base class for Classification models.

        Attributes:
        -----------
            coeff_ : array (float)
                the coefficient (weights) vector

            __is_fitted : bool
                to check if the model is fitted or not

            __summary: dict
                'accuracy': float
                    accuracy of the classifier 
                'cross_entropy': float
                    error of the classifier
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

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):

        X_b = X

        self._fit(X, y)
        self.__is_fitted = True

        n_features = X_b.shape[1]

        self.__summary['accuracy'] = self.accuracy_score(X_b, y)
        self.__summary['cross-entropy'] = self.cross_entropy(X_b, y)
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

    def predict_proba(self, X):
        # add the intercept column to the feature vector
        X = self._add_intercept(X)
        # compute X.W
        return self.sigmoid(X.dot(self.coeff_))

    def _predict(self, X):
        return np.where(self.predict_proba(X) <= 0.5, 0, 1)

    def summary(self):
        print(f"Accuracy: {self.__summary['accuracy']}")
        print(f"Logistic Cost: {self.__summary['cross-entropy']}")
        print("Coefficients: ")
        for s, c in self.__summary['coefficients']:
            print(f"   {s}: {c}")
        print(f"Intercept: {self.__summary['intercept']}")

    def cross_entropy(self, X, y):
        y_pred = self.predict_proba(X)
        err = y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)
        return - np.mean(err)

    def confusion_matrix(self, X, y):
        y_pred = self.predict(X)
        K = len(np.unique(y))

        result = np.zeros((K, K))

        for pred, exp in zip(y_pred, y):
            result[pred][exp] += 1

        return result

    def accuracy_score(self, X, y, cutoff=0.5):
        cf_mat = self.confusion_matrix(X, y)
        acc = np.sum(np.diag(cf_mat)) / np.sum(cf_mat)
        return acc