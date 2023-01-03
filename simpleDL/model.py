import numpy as np
from collections import OrderedDict
from simpleDL.function import *


class BackBone:
    def __init__(self):
        self.network = OrderedDict()


    def __call__(self, arg):
        result = self._forward(arg)
        return result

    
    def _forward():
        pass


class DenseLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.differentiable = True
        self.x = None
        self.parameter = OrderedDict()

        self.parameter["weight"] = 0.01 * np.random.randn(input_size, output_size)
        self.parameter["bias"] = np.zeros(output_size)

        self.dw = None
        self.db = None


    def __call__(self, arg):
        result = self._forward(arg)
        return result

    
    def __repr__(self) -> str:
        return "DenseLayer"

    
    def _forward(self, x):
        self.x = x
        result = np.dot(x, self.parameter["weight"]) + self.parameter["bias"]
        return result

    
    def _backward(self, input):
        dx = np.dot(input, self.parameter["weight"].T)
        self.dw = np.dot(self.x.T, input)
        self.db = np.sum(input, axis=0)

        return dx


    def load_parameter(self, parameter: tuple):
        weight, bias = parameter
        self.parameter["weight"] = np.array(weight)
        self.parameter["bias"] = np.array(bias)


class MakeModel(BackBone):
    def __init__(self, layers: list):
        super().__init__()
        self.sequence = []
        self.grad = OrderedDict()
        dense_count = 1
        function_count = 1
        self.layers = layers
        for layer in self.layers:
            if repr(layer) == "DenseLayer":
                self.network[f"{repr(layer)}{dense_count}"] = layer
                self.sequence.append(f"{repr(layer)}{dense_count}")
                dense_count += 1

            elif repr(layer) == "Function":
                self.network[f"{repr(layer)}{function_count}"] = layer
                self.sequence.append(f"{repr(layer)}{function_count}")
                function_count += 1


    def __call__(self, arg):
        result = self._forward(arg)
        return result

    
    def __str__(self) -> str:
        return "model"


    def _forward(self, x):
        input = x
        for layer_name in self.sequence:
            layer = self.network[layer_name]
            y = layer(input)
            input = y

        return input
        
    
    def _backward(self, loss):
        result = loss._backward()
        for layer_name in reversed(self.sequence):
            layer = self.network[layer_name]
            result = layer._backward(result)

    
    def _update(self, lr= 0.01):
        for layer_name in self.sequence:
            layer = self.network[layer_name]
            if layer.differentiable:
                self.network[layer_name].parameter["weight"] -= (lr * layer.dw)
                self.network[layer_name].parameter["bias"] -= (lr * layer.db)


    def gradient(self):
        count = 1
        for layer_name in self.sequence:
            layer = self.network[layer_name]
            if layer.differentiable:
                self.grad[f"w{count}"] = layer.dw
                self.grad[f"b{count}"] = layer.db

        return self.grad
                


