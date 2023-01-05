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
    def __init__(self, input_size: int, output_size: int, initialize: str = "He"):
        """
        Initialize DenseLayer

        Parameters
        ----------
        input_size (int) : input node 개수

        output_size (int) : output node 개수

        initialize (str, optional) : 가중치 초기화 방법 설정. Default: "He"

        """
        self.input_size = input_size
        self.output_size = output_size
        self.differentiable = True
        self.x = None
        # for 4-dim tensor
        self.orgin_x_shape = None
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
        # for 4-dim tensor
        self.origin_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        result = np.dot(x, self.parameter["weight"]) + self.parameter["bias"]
        return result

    
    def _backward(self, input):
        """
        
        Parameter
        ---------
        input : 다음 layer의 해당 layer의 output에 대한 미분값

        Variable
        --------
        dx = forward시 input에 대한 미분값으로 이전 layer로 넘겨준다.
        self.dw = weight에 대한 미분값
        self.db = bias에 대한 미분값

        """
        dx = np.dot(input, self.parameter["weight"].T)
        self.dw = np.dot(self.x.T, input)
        self.db = np.sum(input, axis=0)

        # for 4-dim tensor
        dx = dx.reshape(*self.origin_x_shape)
        return dx


    def load_parameter(self, parameter: tuple):
        weight, bias = parameter
        self.parameter["weight"] = np.array(weight)
        self.parameter["bias"] = np.array(bias)


class ConvLayer():
    """
    Convolution layer
    """
    def __init__(self, input_channel: int, output_channel: int, kernel_size: int, stride = 1, padding = 0, initialize = "He"):
        """
        Initialize ConvLayer

        Parameters
        ----------
        input_channel (int) : input의 channel 개수

        output_channel (int) : kernel의 개수

        kernel_size (int or tuple) : kernel(height, width)의 크기

        stride (int, optional) : stride 설정. Default: 1

        padding (int, optional) : padding 설정. Default: 0

        initialize (str, optional) : 가중치 초기화 방법 설정. Default: "He"


        """
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
        col = col.reshape(n_input, out_height, out_width, n_input_channel, self.kernel_height, self.kernel_width).transpose(0, 3, 4, 5, 1, 2)

        img = np.zeros((n_input, n_input_channel, input_height + 2 * self.padding + self.stride - 1, input_width + 2 * self.padding + self.stride - 1))
        for y in range(self.kernel_height):
            y_max = y + self.stride * out_height
            for x in range(self.kernel_width):
                x_max = x + self.stride * out_width
                img[:, :, y:y_max:self.stride, x:x_max:self.stride] += col[:, :, y, x, :, :]

        return img[:, :, self.padding:input_height + self.padding, self.padding:input_width + self.padding]


class MaxPoolingLayer():
    def __init__(self, kernel_size, stride = 1, padding = 0):
        self.differentiable = False
        if isinstance(kernel_size, int):
            self.kernel_width = kernel_size
            self.kernel_height = kernel_size

        elif isinstance(kernel_size, tuple):
            self.kernel_height, self.kernel_width = kernel_size

        self.stride = stride
        self.padding = padding

        self.x = None
        self.mask = None

    def __repr__(self) -> str:
        return "MaxPoolingLayer"


    def __call__(self, arg):
        result = self._forward(arg)
        return result


    def _forward(self, x):
        n_input, n_input_channel, input_height, input_width = x.shape

        out_height = int(1 + (input_height + self.padding * 2 - self.kernel_height) / self.stride)
        out_width = int(1 + (input_width + self.padding * 2 - self.kernel_width) / self.stride)

        col = self.img2col(x)
        col = col.reshape(-1, self.kernel_height * self.kernel_width)

        self.x = x
        self.mask = np.argmax(col, axis=1) # (-1, self.kernel_height*kernel_width)
        result = np.max(col, axis=1)
        result = result.reshape(n_input, out_height, out_width, n_input_channel).transpose(0, 3, 1, 2)

        return result

    
    def _backward(self, input):
        input = input.transpose(0, 2, 3, 1) # (n_input, n_input_channel, input_height, input_width) -> (n_input, out_height, out_width, n_input_channel)
        kernel_size = self.kernel_height * self.kernel_width
        dmax = np.zeros((input.size, kernel_size)) # (n_input*n_input_channel*input_height*input_width, self.kernel_height*kernel_width)
        dmax[np.arange(self.mask.size), self.mask.flatten()] = input.flatten()
        dmax = dmax.reshape(input.shape + (kernel_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
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
        col = col.reshape(n_input, out_height, out_width, n_input_channel, self.kernel_height, self.kernel_width).transpose(0, 3, 4, 5, 1, 2)

        img = np.zeros((n_input, n_input_channel, input_height + 2 * self.padding + self.stride - 1, input_width + 2 * self.padding + self.stride - 1))
        for y in range(self.kernel_height):
            y_max = y + self.stride * out_height
            for x in range(self.kernel_width):
                x_max = x + self.stride * out_width
                img[:, :, y:y_max:self.stride, x:x_max:self.stride] += col[:, :, y, x, :, :]

        return img[:, :, self.padding:input_height + self.padding, self.padding:input_width + self.padding]



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
        self.layers = layers

        self.count_dict = OrderedDict()
        temp_repr_list = []

        for layer in self.layers:
            temp_repr_list.append(repr(layer))
        repr_set = set(temp_repr_list)

        #initialize count_dict
        for rep in repr_set:
            self.count_dict[rep] = 1

        for layer in self.layers:
            self.network[f"{repr(layer)}{self.count_dict[repr(layer)]}"] = layer
            self.sequence.append(f"{repr(layer)}{self.count_dict[repr(layer)]}")
            self.count_dict[repr(layer)] += 1

            

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
        if repr(layer) not in self.count_dict.keys():
            self.count_dict[repr(layer)] = 1
            
        self.network[f"{repr(layer)}{self.count_dict[repr(layer)]}"] = layer
        self.sequence.append(f"{repr(layer)}{self.count_dict[repr(layer)]}")
        self.count_dict[repr(layer)] += 1
        
