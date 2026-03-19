import numpy as np
from abc import ABC, abstractmethod


class Activation(ABC):
    is_output = False
    
    @abstractmethod
    def __call__(self, X) -> np.ndarray:
        pass

    @abstractmethod
    def grad(self, X) -> int|np.ndarray:
        pass


class ReLu(Activation):

    def __call__(self, X):
        return np.maximum(X,0)
    
    def grad(self, X):
        if self.is_output: return 1
        return (X >0).astype(int)
    

class Sigmoid(Activation):

    def __call__(self, X, clip=True):
        if clip: X = np.clip(X, -500, 500)
        return 1 / (1 + np.exp(-X))
    
    def grad(self, X):
        if self.is_output: return 1
        return self(X) * (1-self(X))
    

class Softmax(Activation):

    def __call__(self, X):
        shift_X = X - np.max(X, axis=0, keepdims=True)
        exp_X = np.exp(shift_X)
        return exp_X / np.sum(exp_X, axis=0, keepdims=True)
    
    def grad(self, X):
        if self.is_output: return 1
        return X # temp