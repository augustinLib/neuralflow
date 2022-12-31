import numpy as np
from function import *


class BackBone():
    def __init__(self):
        self.network = {}


    def __call__(self, arg):
        result = self.forward(arg)
        return result

    
    def forward():
        pass


class DenseLayer():
    def __init__(self, input_size, output_size, random_init = True):
        self.input_size = input_size
        self.output_size = output_size
        self.parameter = {}
        if random_init:
            self.parameter["weight"] = np.random.random((input_size, output_size))
            self.parameter["bias"] = np.random.random((output_size))
        else:
            self.parameter["weight"] = np.zeros((input_size, output_size))
            self.parameter["bias"] =  np.zeros((output_size))


    def __call__(self, arg):
        result = self.forward(arg)
        return result

    
    def __repr__(self) -> str:
        return "DenseLayer"

    
    def forward(self, x):
        result = np.dot(x, self.parameter["weight"]) + self.parameter["bias"]
        return result


    def load_parameter(self, parameter: tuple):
        weight, bias = parameter
        self.parameter["weight"] = np.array(weight)
        self.parameter["bias"] = np.array(bias)


class MakeModel(BackBone):
    def __init__(self, layers: list):
        super().__init__()
        self.sequence = []
        dense_count = 1
        function_count = 1
        for layer in layers:
            if repr(layer) == "DenseLayer":
                self.network[f"{repr(layer)}{dense_count}"] = layer
                self.sequence.append(f"{repr(layer)}{dense_count}")
                dense_count += 1

            elif repr(layer) == "Function":
                self.network[f"{repr(layer)}{function_count}"] = layer
                self.sequence.append(f"{repr(layer)}{function_count}")
                function_count += 1


    def __call__(self, arg):
        result = self.forward(arg)
        return result


    def forward(self, x):
        input = x
        for layer_name in self.sequence:
            layer = self.network[layer_name]
            y = layer(input)
            input = y

        return input
        
    
    def printmodel(self):
        print(self.network)

