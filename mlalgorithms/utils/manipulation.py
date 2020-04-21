import numpy as np

def fetch_batches(X, y=None, batch_size=32):
    '''
        Function to divide the dataset into batch of size `batch_size`.
        
        Parameters
        ----------
            X: np.array
                set of features
            y: np.array, default=None
                target values
            batch_size: int, default=True
                number of samples in each batch
    '''
    n_samples = X.shape[0]
    # shuffle data before every iteration
    indices = np.random.permutation(np.arange(n_samples))
    X = X[indices]

    if y is not None:
        y = y[indices]

    for i in range(0, n_samples, batch_size):
        begin = i
        end = min(i + batch_size, n_samples)
        if y is not None:
            yield X[begin:end], y[begin:end]
        else:
            yield X[begin:end]

def one_hot_encoding(y, n_classes=None):
    '''
        One-Hot Encoding of classes.
        
        Example,
            y = [1, 2, 0, 1]

            one_hot = [
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0]
            ]
    '''
    if n_classes is None:
        n_classes = len(np.unique(y))
    n_samples = y.shape[0]
    one_hot = np.zeros((n_samples, n_classes))

    one_hot[range(n_samples), y] = 1
    return one_hot, n_classes

def unhot_encoding(y):
    '''
        Convert one-hot encoding back to nominal values.
    '''
    return np.argmax(y)