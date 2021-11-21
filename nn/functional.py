import numpy as np

def sigmoid(x):
    """
    Calculates the sigmoid activation function
    """
    return 1 / (1 + np.exp(-x))

def tanh(x):
    """
    Calculates the tanh activation function
    """
    return np.tanh(x)
