import numpy as np


def reLu(z: np.ndarray) -> np.ndarray:
    """Basic activation function (x if x>0 else 0)"""
    return np.maximum(0,z)


def reLu_deriv(z: np.ndarray) -> np.ndarray:
    """Derivate of reLu function"""
    return (z >0).astype(int)


def softmax(z: np.ndarray) -> np.ndarray:
    """Output activation function for single-label classification
    (probability between 0 and 1 for each output node, sum of nodes' probabilities = 1)"""
    return np.exp(z) / np.sum(np.exp(z))


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Output activation function for multi-label classification
    (probability between 0 and 1 for each output node independently)"""
    return 1 / (1 + np.exp(-z))