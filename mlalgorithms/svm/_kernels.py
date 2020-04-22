import numpy as np



def linear_kernel(**kwargs):
    '''
        Linear Kernel Function

        Parameters
        ----------
            x_i, x_j: np.array
                feature vectors
    '''
    def f(x_i, x_j):
        return np.inner(x_i, x_j)
    return f

def polynomial_kernel(const, power, **kwargs):
    '''
        Polynomial Kernel Function

        Parameters
        ----------
            x_i, x_j: np.array
                feature vectors
            const: float
                constant to add in the kernel function
            power: int
                degree of the kernel function
    '''
    def f(x_i, x_j):
        return (np.inner(x_i, x_j) + const) ** power
    return f

def rbf_kernel(gamma, **kwargs):
    '''
        Radial Basic Function as Kernel Function

        Parameters
        ----------
            x_i, x_j: np.array
                feature vectors
            gamma: float
                provided for the kernel function
    '''
    def f(x_i, x_j):
        distance = np.linalg.norm(x_i - x_j) ** 2
        return np.exp(- gamma * distance)
    return f