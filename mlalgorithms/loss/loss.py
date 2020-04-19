import numpy as np
from abc import abstractmethod

class Loss(object):

    @abstractmethod
    def cost(self, y_true, y_pred):
        pass

    @abstractmethod
    def derivative(self, y_true, y_pred):
        pass

class SquaredLoss(Loss):
    '''
        Mean Squared Loss
            Cost function for regression algorithms.
    '''
    def cost(self, y_true, y_pred):
        return 0.5 * (np.power(y_true - y_pred, 2))
    
    def derivative(self, y_true, y_pred):
        return - (y_true - y_pred)

class CrossEntropy(Loss):
    '''
        Cross Entropy Loss
            Cost function for classification algorithms
    '''
    def cost(self, y_true, y_pred):
        # Avoid division by zero error
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -((y_true * np.log(y_pred)) + ((1 - y_true) * np.log(1 - y_pred)))
    
    def derivative(self, y_true, y_pred):
        # Avoid division by zero error
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -((y_true / y_pred) - ((1 - y_true) / (1 - y_pred)))