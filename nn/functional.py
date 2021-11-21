import numpy as np

def leaky_relu(x):
    """
    Calculates the leaky relu activation function
    """
    return np.maximum(0.01 * x, x)

def relu(x):
    """
    Calculates the relu activation function
    """
    return np.maximum(0, x)

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
