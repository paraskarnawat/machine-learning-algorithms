import numpy as np
from itertools import combinations_with_replacement


def shuffle(X, y, random_state=None):
    '''
        Shuffle the data.

        Parameters
        ----------
            X: np.array
                set of features
            y: np.array
                target attribute
            random_state: int
                seed for random values
    '''
    if random_state:
        np.random.seed(random_state)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    return X[indices], y[indices]

def train_test_split(X, y, test_size=0.3, shuffle_data=True, random_state=None):
    '''
        Split dataset into train and test set

        Parameters
        ----------
            X: np.array
                set of features
            y: np.array
                target attribute
            test_size: float
                fraction of test set
            shuffle: bool
                shuffle the data
            random_state: int
                seed

        Return X_train, X_test, y_train, y_test
    '''
    if shuffle_data:
        X, y = shuffle(X, y, random_state=random_state)
    
    n_rows = X.shape[0]
    split_indices = int(n_rows * test_size)

    X_test, y_test = X[:split_indices], y[:split_indices]
    X_train, y_train = X[split_indices:], y[split_indices:]

    return X_train, X_test, y_train, y_test

def fetch_batches(X, y=None, batch_size=32):
    '''
        Yield mini-batches of data

        Parameters
        ----------
            X: np.array
                set of features
            y: np.array
                target attribute
            batch_size: int
                the size of each batch
                if equal to 1, each row is returned
                if equal to len(X), whole dataset is returned
    '''
    n_rows = X.shape[0]

    for i in np.arange(0, n_rows, batch_size):
        begin, end = i, min(i + batch_size, n_rows)
        if y is not None:
            yield X[begin:end], y[begin:end]
        else:
            yield X[begin:end]


def polynomial_features(X, degree=1):
    '''
        Generate polynomial features of `X`

        For example, if `X` contains features - x1, x2
        and degree is 2, then following features are returned:

            x1^2, x1.x2, x2^2

        Parameters
        ----------
            X: np.array
                set of features
            degree: int 
                highest degree of polynomial

        Returns the transformed features.
    '''
    n_rows, n_cols = X.shape

    def index_combinations():
        combs = [combinations_with_replacement(
            range(n_cols), i) for i in range(0, degree + 1)]
        flat = [item for sublist in combs for item in sublist]
        return flat[1:]

    combinations = index_combinations()
    n_output_col = len(combinations)

    poly_X = np.empty((n_rows, n_output_col))

    for i, indices in enumerate(combinations):
        poly_X[:, i] = np.prod(X[:, indices], axis=1)

    return poly_X


def normalize(X):
    '''
        Normalize the features where mean = 0, standard deviation = 1

        Parameters
        ----------
            X: np.array
                set of features

        Returns the normalized set of features
    '''
    X_mean = np.mean(X, axis=0)
    X_dev = np.std(X, axis=0)

    # Avoid division by zero error
    X_dev[X_dev == 0] = 1

    X_normalized = np.subtract(X, X_mean)

    return X_normalized / X_dev


def one_hot_encoding(y, n_labels=None):
    '''
        One-hot encoding of nominal values
        Example:

            y = [0, 2, 1, 0]

            one_hot = [
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0]
            ]
    '''
    if n_labels is None:
        n_labels = np.amax(y) + 1

    one_hot = np.zeros(y.shape[0], n_labels)
    one_hot[np.arange(y.shape[0]), y] = 1

    return one_hot


def nominal(y):
    '''
        Convert one-hot encoding back to nominal value
        Example:
            y = [
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0]
            ]

            nominal = [0, 2, 1, 0]
    '''
    return np.argmax(y, axis=1)
