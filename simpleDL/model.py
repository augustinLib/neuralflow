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


class DenseLayer():
    def __init__(self, input_size, output_size, initialize = "He"):
        self.input_size = input_size
        self.output_size = output_size
        self.differentiable = True
        self.x = None
        self.parameter = OrderedDict()

        if initialize == "He":
            self.parameter["weight"] = np.random.randn(input_size, output_size) * (np.sqrt(2 / input_size))
            self.parameter["bias"] = np.zeros(output_size)    

        elif initialize == "Xavier":
            self.parameter["weight"] = np.random.randn(input_size, output_size) * np.sqrt(input_size)
            self.parameter["bias"] = np.zeros(output_size)

        elif initialize == "None":
            self.parameter["weight"] = 0.01 * np.random.randn(input_size, output_size)
            self.parameter["bias"] = np.zeros(output_size)

        else:
            raise ValueError("'initialize' must be 'He' or 'Xavier' or 'None'")


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


class ConvLayer():
    def __init__(self, input_channel, output_channel, kernel_size, stride = 1, padding = 0, initialize = "He"):
        self.differentiable = True
        self.parameter = OrderedDict()
        self.input_channel = input_channel
        self.output_channel = output_channel

        if isinstance(kernel_size, int):
            self.kernel_width = kernel_size
            self.kernel_height = kernel_size

        elif isinstance(kernel_size, tuple):
            self.kernel_height, self.kernel_width = kernel_size

        self.stride = stride
        self.padding = padding
        self.fan_in = kernel_size * kernel_size * input_channel
        self.fan_out = kernel_size * kernel_size * output_channel

        self.x = None
        self.col = None
        self.col_weight = None
        self.dw = None
        self.db = None
        


        if initialize == "He":
            self.parameter["weight"] = np.random.randn(self.output_channel, self.input_channel, self.kernel_height, self.kernel_width) * (np.sqrt(2 / self.fan_in))
            self.parameter["bias"] = np.zeros(self.output_channel)    

        elif initialize == "Xavier":
            self.parameter["weight"] = np.random.randn(self.output_channel, self.input_channel, self.kernel_height, self.kernel_width) * np.sqrt(self.fan_in)
            self.parameter["bias"] = np.zeros(self.output_channel)

        elif initialize == "None":
            self.parameter["weight"] = 0.01 * np.random.randn(self.output_channel, self.input_channel, self.kernel_height, self.kernel_width)
            self.parameter["bias"] = np.zeros(self.output_channel)

        else:
            raise ValueError("'initialize' must be 'He' or 'Xavier' or 'None'")


    def __repr__(self) -> str:
        return "ConvLayer"


    def __call__(self, arg):
        result = self._forward(arg)
        return result


    def _forward(self, x):
        n_input, n_input_channel, input_height, input_width = x.shape
        out_height = int(1 + (input_height + self.padding * 2 - self.kernel_height) / self.stride)
        out_width = int(1 + (input_width + self.padding * 2 - self.kernel_width) / self.stride)

        col = self.img2col(x)
        col_weight = self.parameter["weight"].reshape(self.output_channel, -1).T
        result = np.dot(col, col_weight) + self.parameter["bias"]
        result = result.reshape(n_input, out_height, out_width, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_weight = col_weight

        return result

    
    def _backward(self, input):
        input = input.transpose(0,2,3,1).reshape(-1, self.output_channel)

        self.dw = np.dot(self.col.T, input)
        self.dw = self.dw.transpose(1,0).reshape(self.output_channel, self.input_channel, self.kernel_height, self.kernel_width)
        self.db = np.sum(input, axis=0)

        dcol = np.dot(input, self.col_weight.T)
        result = self.col2img(dcol, self.x.shape)

        return result


    def img2col(self, input_data):
        n_input, n_input_channel, input_height, input_width = input_data.shape
        out_height = (input_height + self.padding * 2 - self.kernel_height) // self.stride + 1
        out_width = (input_width + self.padding * 2 -self.kernel_width) // self.stride + 1

        img = np.pad(input_data, [(0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)], 'constant')
        col = np.zeros((n_input, n_input_channel, self.kernel_height, self.kernel_width, out_height, out_width))

        for y in range(self.kernel_height):
            y_max = y + self.stride * out_height
            for x in range(self.kernel_width):
                x_max = x + self.stride * out_width
                col[:, :, y, x, :, :] = img[:, :, y:y_max:self.stride, x:x_max:self.stride]

        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(n_input * out_height * out_width, -1)
        return col


    def col2img(self, col, input_shape):
        n_input, n_input_channel, input_height, input_width = input_shape
        out_height = (input_height + 2 * self.padding - self.kernel_height) // self.stride + 1
        out_width = (input_width + 2 * self.padding - self.kernel_width) // self.stride + 1
        col = col.reshape(n_input, out_height, out_width, C, self.kernel_height, self.kernel_width).transpose(0, 3, 4, 5, 1, 2)

        img = np.zeros((n_input, n_input_channel, input_height + 2 * self.padding + self.stride - 1, input_width + 2 * self.padding + self.stride - 1))
        for y in range(self.kernel_height):
            y_max = y + self.stride * out_height
            for x in range(self.kernel_width):
                x_max = x + self.stride * out_width
                img[:, :, y:y_max:self.stride, x:x_max:self.stride] += col[:, :, y, x, :, :]

        return img[:, :, self.padding:input_height + self.padding, self.padding:input_width + self.padding]


class MaxPoolingLayer():
    def __init__(self):
        self.differentiable = False


    def __repr__(self) -> str:
        return "MaxPoolingLayer"


    def __call__(self, arg):
        result = self._forward(arg)
        return result


    def _forward(self, x):
        n_input, n_input_channel, input_height, input_width = x.shape

        out_height = int(1 + (input_height + self.padding * 2 - self.kernel_size) / self.stride)
        out_width = int(1 + (input_width + self.padding * 2 - self.kernel_size) / self.stride)

        col = self.img2col(x)
        weight_col = self.parameter["weight"].reshape(self.output_channel, -1).T
        result = np.dot(col, weight_col) + self.parameter["bias"]
        result = result.reshape(n_input, out_height, out_width, -1).transpose(0, 3, 1, 2)

        return result

    
    def _backward(self, input):


        return None    



class BatchNorm():
    def __init__(self, epsilon = 1e-8):
        self.differentiable = True
        self.mean = None
        self.std = None
        self.parameter = OrderedDict()



    def _forward(self, x):
        self.x = x
        result = np.dot(x, self.parameter["weight"]) + self.parameter["bias"]
        return result

    def __repr__(self) -> str:
        return "BatchNormLayer"



class Model(BackBone):
    def __init__(self, *layers):
        super().__init__()
        self.sequence = []
        self.grad = OrderedDict()
        self.dense_count = 1
        self.function_count = 1
        self.conv_count = 1
        self.layers = layers
        for layer in self.layers:
            if repr(layer) == "DenseLayer":
                self.network[f"{repr(layer)}{self.dense_count}"] = layer
                self.sequence.append(f"{repr(layer)}{self.dense_count}")
                self.dense_count += 1

            elif repr(layer) == "Function":
                self.network[f"{repr(layer)}{self.function_count}"] = layer
                self.sequence.append(f"{repr(layer)}{self.function_count}")
                self.function_count += 1

            elif repr(layer) == "ConvLayer":
                self.network[f"{repr(layer)}{self.conv_count}"] = layer
                self.sequence.append(f"{repr(layer)}{self.conv_count}")
                self.conv_count += 1


    def __call__(self, arg):
        result = self._forward(arg)
        return result

    
    def __str__(self) -> str:
        structure = ""
        string_list = []
        for i, layer_name in enumerate(self.sequence):
            layer = self.network[layer_name]

            if layer.differentiable:
                shape = layer.parameter["weight"].shape
                string_list.append(f"{i}. {layer_name} : {layer} {shape} \n")
            else:
                string_list.append(f"{i}. {layer_name} : {layer}\n")

        structure = structure.join(string_list)
        return structure
            


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
                

    def add_layer(self, layer):
        if repr(layer) == "DenseLayer":
            self.network[f"{repr(layer)}{self.dense_count}"] = layer
            self.sequence.append(f"{repr(layer)}{self.dense_count}")
            self.dense_count += 1

        elif repr(layer) == "Function":
            self.network[f"{repr(layer)}{self.function_count}"] = layer
            self.sequence.append(f"{repr(layer)}{self.function_count}")
            self.function_count += 1

