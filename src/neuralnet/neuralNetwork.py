from neuralnet.layers import *
from neuralnet.activations import *

    
class NeuralNetwork:

    def __init__(self, layers, alpha, momentum) -> None:
        self.alpha = alpha
        self.momentum = momentum
        self.layers = [layer(type, in_size, out_size, activation) for type,in_size,out_size,activation in layers.values()]
        self.layers[-1].activation.is_output = True

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        self.output = X

    def backward(self, targets, update_weigths = True):
        delta = self.output - targets
        for layer in reversed(self.layers):
            delta = layer.backward(delta, self.alpha, update_weigths, momentum=self.momentum)
    
    
