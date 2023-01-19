import numpy as np
from collections import OrderedDict
from simpleDL.function import *
from copy import deepcopy


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
            self.parameter["weight"] = np.random.randn(input_size, output_size) * np.sqrt(1/input_size)
            self.parameter["bias"] = np.zeros(output_size)

        elif initialize == "None":
            self.parameter["weight"] = 0.01 * np.random.randn(input_size, output_size)
            self.parameter["bias"] = np.zeros(output_size)

        else:
            raise ValueError("'initialize' must be 'He' or 'Xavier' or 'None'")


        self.dw = np.zeros_like(self.parameter["weight"])
        self.db = np.zeros_like(self.parameter["bias"])


    def __call__(self, arg):
        result = self._forward(arg)
        return result

    
    def __repr__(self):
        return "DenseLayer"

    
    def _forward(self, x):
        if x.ndim == 3:
            batch_size, n_timestep, _ = x.shape
            reshaped_x = x.reshape(batch_size * n_timestep, -1)
            self.x = x
            result = np.matmul(reshaped_x, self.parameter["weight"]) + self.parameter["bias"]
            result = result.reshape(batch_size, n_timestep, -1)

            return result

        self.origin_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        result = np.matmul(x, self.parameter["weight"]) + self.parameter["bias"]

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
        if self.x.ndim == 3:
            x = self.x
            batch_size, n_timestep, _ = x.shape
            input = input.reshape(batch_size * n_timestep, -1)
            reshaped_x = x.reshape(batch_size * n_timestep, -1)

            db = np.sum(input, axis=0)
            dw = np.matmul(reshaped_x.T, input)
            dx = np.matmul(input, self.parameter["weight"].T)
            dx = dx.reshape(*x.shape)
            
            self.dw[...] = dw
            self.db[...] = db

            return dx
        
        else:
            dx = np.matmul(input, self.parameter["weight"].T)
            self.dw = np.matmul(self.x.T, input)
            self.db = np.sum(input, axis=0)

            # for 4-dim tensor
            dx = dx.reshape(*self.origin_x_shape)

            return dx


    def load_parameter(self, parameter: tuple):
        weight, bias = parameter
        self.parameter["weight"] = np.array(weight)
        self.parameter["bias"] = np.array(bias)


    def get_gradient(self):
        grad = OrderedDict()
        grad["dw"] = self.dw
        grad["db"] = self.db

        return grad


class Embedding():
    def __init__(self, parameter):
        """
        Initialize EmbeddingLayer

        Parameters
        ----------
        vocab_size (int) : vocab의 size

        hidden_size (int) : embedding될 representation의 hidden size

        initialize (str, optional) : 가중치 초기화 방법 설정. Default: "He"

        """
        self.differentiable = True
        self.index = None
        self.parameter = OrderedDict()
        self.parameter["weight"] = parameter["weight"]
        self.dw = np.zeros_like(self.parameter["weight"])


    def __call__(self, arg):
        result = self._forward(arg)
        return result

    
    def __repr__(self):
        return "Embedding"

    
    def _forward(self, index):
        self.index = index
        weight = self.parameter["weight"]
        result = weight[index]
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
        dw = self.dw
        dw[...] = 0
        np.add.at(dw, self.index, input)

        return None


    def _get_gradient(self):
        dw = self.dw

        return dw


class EmbeddingLayer():
    def __init__(self, vocab_size: int, hidden_size: int, initialize = "He"):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.differentiable = True
        self.parameter = OrderedDict()

        if initialize == "He":
            self.parameter["weight"] = np.random.randn(vocab_size, hidden_size).astype(np.float32) * (np.sqrt(2 / vocab_size))


        elif initialize == "Xavier":
            self.parameter["weight"] = np.random.randn(vocab_size, hidden_size).astype(np.float32) * np.sqrt(1/vocab_size)


        elif initialize == "None":
            self.parameter["weight"] = 0.01 * np.random.randn(vocab_size, hidden_size).astype(np.float32)

        else:
            raise ValueError("'initialize' must be 'He' or 'Xavier' or 'None'")

        self.layer = None
        self.dw = np.zeros_like(self.parameter["weight"]).astype(np.float32)


    def __call__(self, arg):
        result = self._forward(arg)
        return result


    def __repr__(self):
        return "EmbeddingLayer"


    def _forward(self, x):
        batch_size, n_timestep = x.shape

        result = np.empty((batch_size, n_timestep, self.hidden_size)).astype(np.float32)
        self.layer = []
        
        for timestep in range(n_timestep):
            embedding_cell = Embedding(self.parameter)
            result[:, timestep, :] = embedding_cell._forward(x[:, timestep])
            self.layer.append(embedding_cell)

        return result

    def _backward(self, dout):
        w = self.parameter["weight"]
        batch_size, n_timestep, hidden_size = dout.shape

        dw = np.zeros_like(w).astype(np.float32)
        for timestep in range(n_timestep):
            embedding_cell = self.layer[timestep]
            embedding_cell._backward(dout[:, timestep, :])
            temp_dw = embedding_cell._get_gradient()
            dw += temp_dw
        
        self.dw[...] = dw

        return None

    def get_gradient(self):
        grad = OrderedDict()
        grad["dw"] = self.dw

        return grad


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
            self.parameter["weight"] = np.random.randn(self.output_channel, self.input_channel, self.kernel_height, self.kernel_width).astype(np.float32) * (np.sqrt(2 / self.fan_in))
            self.parameter["bias"] = np.zeros(self.output_channel).astype(np.float32) 

        elif initialize == "Xavier":
            self.parameter["weight"] = np.random.randn(self.output_channel, self.input_channel, self.kernel_height, self.kernel_width).astype(np.float32) * np.sqrt(1/ self.fan_in)
            self.parameter["bias"] = np.zeros(self.output_channel).astype(np.float32)

        elif initialize == "None":
            self.parameter["weight"] = 0.01 * np.random.randn(self.output_channel, self.input_channel, self.kernel_height, self.kernel_width).astype(np.float32)
            self.parameter["bias"] = np.zeros(self.output_channel).astype(np.float32)

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
        result = np.matmul(col, col_weight) + self.parameter["bias"]
        result = result.reshape(n_input, out_height, out_width, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_weight = col_weight

        return result

    
    def _backward(self, input):
        input = input.transpose(0,2,3,1).reshape(-1, self.output_channel)

        self.dw = np.matmul(self.col.T, input)
        self.dw = self.dw.transpose(1,0).reshape(self.output_channel, self.input_channel, self.kernel_height, self.kernel_width)
        self.db = np.sum(input, axis=0)

        dcol = np.matmul(input, self.col_weight.T)
        result = self.col2img(dcol, self.x.shape)

        return result


    def get_gradient(self):
        grad = OrderedDict()
        grad["dw"] = deepcopy(self.dw)
        grad["db"] = deepcopy(self.db)

        return grad


    def img2col(self, input_data):
        n_input, n_input_channel, input_height, input_width = input_data.shape
        out_height = (input_height + self.padding * 2 - self.kernel_height) // self.stride + 1
        out_width = (input_width + self.padding * 2 -self.kernel_width) // self.stride + 1

        img = np.pad(input_data, [(0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)], 'constant')
        col = np.zeros((n_input, n_input_channel, self.kernel_height, self.kernel_width, out_height, out_width)).astype(np.float32)

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

        img = np.zeros((n_input, n_input_channel, input_height + 2 * self.padding + self.stride - 1, input_width + 2 * self.padding + self.stride - 1)).astype(np.float32)
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
        dmax = np.zeros((input.size, kernel_size)).astype(np.float32) # (n_input*n_input_channel*input_height*input_width, self.kernel_height*kernel_width)
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
        col = np.zeros((n_input, n_input_channel, self.kernel_height, self.kernel_width, out_height, out_width)).astype(np.float32)

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

        img = np.zeros((n_input, n_input_channel, input_height + 2 * self.padding + self.stride - 1, input_width + 2 * self.padding + self.stride - 1)).astype(np.float32)
        for y in range(self.kernel_height):
            y_max = y + self.stride * out_height
            for x in range(self.kernel_width):
                x_max = x + self.stride * out_width
                img[:, :, y:y_max:self.stride, x:x_max:self.stride] += col[:, :, y, x, :, :]

        return img[:, :, self.padding:input_height + self.padding, self.padding:input_width + self.padding]


class RNNCell():
    def __init__(self, parameter):
        self.parameter = OrderedDict()
        self.differentiable = True    
        self.parameter["weight_x"] = parameter["weight_x"]
        self.parameter["weight_h"] = parameter["weight_h"]
        self.parameter["bias"] = parameter["bias"] 
        self.dx = None
        self.dwx = None
        self.dwh = None
        self.db = None
        self.cache = None


    def __repr__(self):
        return "RNNCell"


    def __call__(self, *arg):
        result = self._forward(*arg)
        return result

    
    def _forward(self, x, h_t_prev):
        # (batch_size, hidden_size) x (hidden_size, hidden_size) + (batch_size, input_dim) x (input_dim, hidden_size)
        # => (batch_size, hidden_size)
        temp_t = np.matmul(h_t_prev, self.parameter["weight_h"]) + np.matmul(x, self.parameter["weight_x"]) + self.parameter["bias"] 
        result_t = np.tanh(temp_t)
        # self.cache에 현재 timestep에서의 input, 이전 timestep에서의 hidden state, 현재 timestep에서의 output 저장
        self.cache = x, h_t_prev, result_t

        # (batch_size, hidden_size)
        return result_t


    def _backward(self, input):
        # self.cache에 저장된 현재 timestep에서의 input, 이전 timestep에서의 hidden state, 현재 timestep에서의 output 불러오기
        x, h_t_prev, result_t = self.cache
        # dtanh = 1 - tanh(x)^2
        dtanh = input * (1 - result_t ** 2)
        self.db = np.sum(dtanh, axis=0)
        self.dwh = np.matmul(h_t_prev.T, dtanh)
        self.dwx = np.matmul(x.T, dtanh)
        h_result = np.matmul(dtanh, self.parameter["weight_h"].T)
        x_result = np.matmul(dtanh, self.parameter["weight_x"].T)
        self.dx = x_result

        return x_result, h_result

    
    def _get_gradient(self):
        dx = self.dx
        dwx = self.dwx
        dwh = self.dwh
        db = self.db

        return dx, dwx, dwh, db


class RNNLayer():
    def __init__(self, input_size, hidden_size, n_layers = 1, bidirectional = True, initialize = "He"):
        self.differentiable = True
        self.parameter = OrderedDict()
        self.input_size = input_size
        self.hidden_size = hidden_size

        if initialize == "He":
            self.parameter["weight_x"] = np.random.randn(input_size, hidden_size).astype(np.float32) * (np.sqrt(2 / input_size))
            self.parameter["weight_h"] = np.random.randn(hidden_size, hidden_size).astype(np.float32) * (np.sqrt(2 / hidden_size))
            self.parameter["bias"] = np.zeros(hidden_size).astype(np.float32)    

        elif initialize == "Xavier":
            self.parameter["weight_x"] = np.random.randn(input_size, hidden_size).astype(np.float32) * np.sqrt(1/input_size)
            self.parameter["weight_h"] = np.random.randn(hidden_size, hidden_size).astype(np.float32) * np.sqrt(1/hidden_size)
            self.parameter["bias"] = np.zeros(hidden_size).astype(np.float32)

        elif initialize == "None":
            self.parameter["weight_x"] = 0.01 * np.random.randn(input_size, hidden_size).astype(np.float32)
            self.parameter["weight_h"] = 0.01 * np.random.randn(hidden_size, hidden_size).astype(np.float32)
            self.parameter["bias"] = np.zeros(hidden_size).astype(np.float32)

        self.h = None
        self.dh = None
        self.layer = None
        self.stateful = "stateful"
        self.dx = None
        self.dwx = None
        self.dwh = None
        self.db = None
        self.temp = None


    def __repr__(self):
        return "RNNLayer"


    def __call__(self, arg):
        result = self._forward(arg)
        return result
        
    
    def _forward(self, x):
        wx, wh, b = self.parameter["weight_x"], self.parameter["weight_h"], self.parameter["bias"]
        batch_size, n_timestep, input_dim = x.shape
        input_dim, hidden_size = wx.shape
        
        hidden_state = np.empty((batch_size, n_timestep, hidden_size)).astype(np.float32)
        self.layer = []

        if not self.stateful or self.h is None:
            self.h = np.zeros((batch_size, hidden_size)).astype(np.float32)
        
        for timestep in range(n_timestep):
            rnn_cell = RNNCell(self.parameter)
            # self.h : (batch_size, 1, hidden_size)
            self.h = rnn_cell(x[:, timestep, :], self.h)
            hidden_state[:, timestep, :] = self.h
            self.layer.append(rnn_cell)

        return hidden_state

    
    def _backward(self, dh):
        wx, wh, b = self.parameter["weight_x"], self.parameter["weight_h"], self.parameter["bias"]
        batch_size, n_timestep, hidden_size  = dh.shape
        input_dim, hidden_size = wx.shape
        
        dx = np.empty((batch_size, n_timestep, input_dim)).astype(np.float32)
        
        dwx = np.zeros_like(wx).astype(np.float32)
        dwh = np.zeros_like(wh).astype(np.float32)
        db = np.zeros_like(b).astype(np.float32)

        dh_t = 0

        for timestep in reversed(range(n_timestep)):
            rnn_cell = self.layer[timestep]
            dx_t, dh_t = rnn_cell._backward(dh[:, timestep, :] + dh_t)
            dx[:, timestep, :] = dx_t

            _, temp_dwx, temp_dwh, temp_db = rnn_cell._get_gradient()
            dwx += temp_dwx
            dwh += temp_dwh
            db += temp_db

        self.dwx = dwx
        self.dwh = dwh
        self.db = db
        self.dh = dh_t
        
        # input으로 backpropagation result 전달
        return dx


    def get_gradient(self):
        grad = OrderedDict()
        grad["dwx"] = self.dwx
        grad["dwh"] = self.dwh
        grad["db"] = self.db

        return grad
    
    
    def load_state(self, h):
        self.h = h


    def reset_state(self):
        self.h = None
        
        
class LSTMCell():
    def __init__(self, parameter):
        self.differentiable = True
        self.parameter = OrderedDict()
        self.parameter["weight_x"] = parameter["weight_x"]
        self.parameter["weight_h"] = parameter["weight_h"]
        self.parameter["bias"] = parameter["bias"]
        self.cache = None
        self.dx = None
        self.dwx = None
        self.dwh = None
        self.db = None
        
        
    def __call__(self, *arg):
        result = self._forward(*arg)
        return result
        
        
    def _forward(self, x, h_t_prev, c_t_prev):
        wx, wh, b = self.parameter["weight_x"], self.parameter["weight_h"], self.parameter["bias"]
        batch_size, hidden_size = h_t_prev.shape
        
        a = np.matmul(x, wx) + np.matmul(h_t_prev, wh) + b
        f_temp = a[:, :hidden_size]
        g_temp = a[:, hidden_size:hidden_size*2]
        i_temp = a[:, hidden_size*2:hidden_size*3]
        o_temp = a[:, hidden_size*3:]
        
        f_result = sigmoid(f_temp)
        g_result = np.tanh(g_temp)
        i_result = sigmoid(i_temp)
        o_result = sigmoid(o_temp)
        
        c_t = f_result * c_t_prev + g_result * i_result
        h_t = o_result * np.tanh(c_t)
        
        self.cache = (x, h_t_prev, c_t_prev, f_result, g_result, i_result, o_result, c_t)
        
        return h_t, c_t
    
    
    def _backward(self, dh_t_next, dc_t_next):
        wx, wh, b = self.parameter["weight_x"], self.parameter["weight_h"], self.parameter["bias"]
        x, h_t_prev, c_t_prev, f_result, g_result, i_result, o_result, c_t = self.cache
        
        tanh_c_t = np.tanh(c_t)
        ds = dc_t_next + (dh_t_next * o_result) * (1-tanh_c_t**2)
        do = dh_t_next * tanh_c_t
        di = ds * g_result
        dg = ds * i_result
        df = ds * c_t_prev
        
        ddo = do * o_result * (1-o_result)
        ddi = di * i_result * (1-i_result)
        ddg = dg * g_result * (1-g_result**2)
        ddf = df * f_result * (1-f_result)
        
        da = np.hstack((ddf, ddg, ddi, ddo))
        
        dc_t = ds * f_result

        self.dwx = np.matmul(x.T, da)
        self.dwh = np.matmul(h_t_prev.T, da)
        self.db = da.sum(axis=0)
        
        dx = np.matmul(da, wx.T)
        self.dx = dx
        
        dh_t = np.matmul(da, wh.T)
        
        return dx, dh_t, dc_t
    

    def _get_gradient(self):
        dx = deepcopy(self.dx)
        dwx = deepcopy(self.dwx)
        dwh = deepcopy(self.dwh)
        db = deepcopy(self.db)

        return dx, dwx, dwh, db
    
    
class LSTMLayer():
    def __init__(self, input_size, hidden_size, n_layers = 1, bidirectional = True, initialize = "He"):
        self.differentiable = True
        self.parameter = OrderedDict()
        self.input_size = input_size
        self.hidden_size = hidden_size

        if initialize == "He":
            self.parameter["weight_x"] = np.random.randn(input_size, 4*hidden_size).astype(np.float32) * (np.sqrt(2 / input_size))
            self.parameter["weight_h"] = np.random.randn(hidden_size, 4*hidden_size).astype(np.float32) * (np.sqrt(2 / hidden_size))
            self.parameter["bias"] = np.zeros(4*hidden_size).astype(np.float32)    

        elif initialize == "Xavier":
            self.parameter["weight_x"] = np.random.randn(input_size, 4*hidden_size).astype(np.float32) * np.sqrt(1/input_size)
            self.parameter["weight_h"] = np.random.randn(hidden_size, 4*hidden_size).astype(np.float32) * np.sqrt(1/hidden_size)
            self.parameter["bias"] = np.zeros(4*hidden_size).astype(np.float32)

        elif initialize == "None":
            self.parameter["weight_x"] = 0.01 * np.random.randn(input_size, 4*hidden_size).astype(np.float32)
            self.parameter["weight_h"] = 0.01 * np.random.randn(hidden_size, 4*hidden_size).astype(np.float32)
            self.parameter["bias"] = np.zeros(4*hidden_size).astype(np.float32)

        self.h = None
        self.c = None
        self.dh = None
        self.layer = None
        self.stateful = "stateful"
        self.dx = None
        self.dwx = None
        self.dwh = None
        self.db = None
        self.temp = None
        
        
    def __repr__(self):
        return "LSTMLayer"
    
    
    def __call__(self, *args):
        result = self._forward(*args)
        return result
    
    
    def _forward(self, x):
        wx, wh, b = self.parameter["weight_x"], self.parameter["weight_h"], self.parameter["bias"]
        batch_size, n_timestep, input_dim = x.shape
        hidden_size = wh.shape[0]
        
        self.layer = []
        hidden_state = np.empty((batch_size, n_timestep, hidden_size)).astype(np.float32)
        
        if not self.stateful or self.h is None:
            self.h = np.zeros((batch_size, hidden_size)).astype(np.float32)
        if not self.stateful or self.c is None:
            self.c = np.zeros((batch_size, hidden_size)).astype(np.float32)
        
        for timestep in range(n_timestep):
            lstm_cell = LSTMCell(self.parameter)
            self.h, self.c = lstm_cell(x[:, timestep, :], self.h, self.c)
            hidden_state[:, timestep, :] = self.h
            
            self.layer.append(lstm_cell)
            
        return hidden_state
    
    
    def _backward(self, dh):
        wx, wh, b = self.parameter["weight_x"], self.parameter["weight_h"], self.parameter["bias"]
        batch_size, n_timestep, hidden_size = dh.shape
        input_dim = wx.shape[0]
        
        dx = np.empty((batch_size, n_timestep, input_dim)).astype(np.float32)
        dh_t, dc_t = 0, 0
        
        dwx = np.zeros_like(wx).astype(np.float32)
        dwh = np.zeros_like(wh).astype(np.float32)
        db = np.zeros_like(b).astype(np.float32)
        
        for timestep in reversed(range(n_timestep)):
            lstm_cell = self.layer[timestep]
            dx_t, dh_t, dc_t = lstm_cell._backward(dh[:, timestep, :] + dh_t, dc_t)
            dx[:, timestep, :] = dx_t
            
            _, temp_dwx, temp_dwh, temp_db = lstm_cell._get_gradient()
            dwx += temp_dwx
            dwh += temp_dwh
            db += temp_db
        
        self.dwx = deepcopy(dwx)
        self.dwh = deepcopy(dwh)
        self.db = deepcopy(db)
        self.dh = dh_t
        
        return dx
    
    def get_gradient(self):
        grad = OrderedDict()
        grad["dwx"] = deepcopy(self.dwx)
        grad["dwh"] = deepcopy(self.dwh)
        grad["db"] = deepcopy(self.db)

        return grad
    
    
    def load_state(self, h, c=None):
        self.h, self.c = h, c
    
    
    def reset_state(self):
        self.h, self.c = None, None   
    


class BatchNorm():
    def __init__(self, epsilon = 1e-8):
        self.differentiable = True
        self.mean = None
        self.std = None
        self.parameter = OrderedDict()


    def _forward(self, x):
        self.x = x
        result = np.matmul(x, self.parameter["weight"]) + self.parameter["bias"]
        return result


    def __repr__(self):
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

    
    def __str__(self):
        structure = ""
        string_list = []
        for i, layer_name in enumerate(self.sequence):
            layer = self.network[layer_name]

            if layer.differentiable:
                if repr(layer) == "RNNLayer":
                    shape = layer.parameter["weight_x"].shape
                    string_list.append(f"{i}. {layer_name} : {layer} {shape} \n")
                    
                elif repr(layer) == "LSTMLayer":
                    shape = layer.parameter["weight_x"].shape
                    lstm_in, lstm_out = shape
                    lstm_out /= 4
                    shape = (lstm_in, int(lstm_out))
                    string_list.append(f"{i}. {layer_name} : {layer} {shape} \n")
                else:
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
        

    def to_gpu(self):
        pass