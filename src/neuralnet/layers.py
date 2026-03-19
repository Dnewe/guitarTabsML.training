import numpy as np
from neuralnet.activations import *
from abc import ABC, abstractmethod


def layer(type, in_size, out_size, activation):
    match activation:
        case "relu":
            activation = ReLu()
        case "sigmoid":
            activation = Sigmoid()
        case "Softmax":
            activation = Softmax()
        case _:
            raise NotImplementedError
    match type:
        case "fc":
            return FCLayer(int(in_size), int(out_size), activation)
        case _:
            raise NotImplementedError


class Layer(ABC):
    activation: Activation
    in_size: int
    out_size: int
    A: np.ndarray
    Z: np.ndarray
    W: np.ndarray

    def __init__(self, in_size, out_size, activation) -> None:
        self.activation = activation
        self.in_size = in_size
        self.out_size = out_size
        self.prev_dW = 0
        self.init_weigths()

    @abstractmethod
    def forward(self, X:np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def init_weigths(self):
        pass
    
    def update(self, dW, alpha, momentum):
        dW += momentum * self.prev_dW
        self.W -= alpha * dW
        self.prev_dW = dW


class FCLayer(Layer):

    def init_weigths(self):
        self.W = np.random.randn(self.in_size +1, self.out_size) * np.sqrt(2.0 / self.in_size)  # +1 for bias

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = np.hstack((X, np.ones((X.shape[0], 1))))     # (batch_size, prev_layer_size +1 (bias))
        self.A = np.dot(self.X, self.W)     # (n_features, layer_size)
        self.Z = self.activation(self.A)
        return self.Z
    
    def backward(self, delta, alpha, update_weigths, momentum=0.9): 
        if self.activation.is_output:
            delta_act = delta
        else:
            delta_act = delta * self.activation.grad(self.A)
        dw = np.dot(self.X.T, delta_act) / delta_act.shape[0]    
        delta_prev = delta_act.dot(self.W.T)[:,:-1] # remove bias column
        if update_weigths:
            self.update(dw, alpha, momentum)
        return delta_prev