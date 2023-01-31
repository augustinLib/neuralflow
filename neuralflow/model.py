import pickle
from copy import deepcopy
from collections import OrderedDict
from neuralflow.function import *
from neuralflow.function_class import *
from neuralflow.gpu import *    
from neuralflow.utils import *


class BaseModel:
    def __init__(self):
        """Base object for Model
        """
        self.network = OrderedDict()


    def __call__(self, arg):
        result = self.forward(arg)
        return result

    
    def forward():
        pass
    

class BaseLayer():
    def __init__(self):
        """Base object for Layer
        """
        self.differentiable = False
        self.changeability = False
        self.mixed_precision = False
        self.tying = False
        self.tied = False
        self.parameter = OrderedDict()

    
    def _to_cpu(self):
        """Layer의 parameter를 vram에서 ram으로 옮김
        """
        param_list = list(self.parameter.keys())
        for param in param_list:
            self.parameter[param] = to_cpu(self.parameter[param])
            
    
    def _to_gpu(self):
        """Layer의 parameter를 ram에서 vram으로 옮김
        """
        param_list = list(self.parameter.keys())
        for param in param_list:
            self.parameter[param] = to_gpu(self.parameter[param])
    
    
    def _save_params(self):
        """Layer의 parameter 반환

        Returns:
            OrderedDict
        """
        return deepcopy(self.parameter)
    
    
    def _load_params(self, params):
        self.parameter = deepcopy(params)
        
    
    def _mixed_precision_training(self):
        self.mixed_precision = True
    
    
    def _mixed_precision_training_off(self):
        self.mixed_precision = False
    

class DenseLayer(BaseLayer):
    def __init__(self, input_size: int, output_size: int, initialize: str = "He"):
        """Initialize DenseLayer

        Parameters
        ----------
        input_size (int) : input node 개수

        output_size (int) : output node 개수

        initialize (str, optional) : 가중치 초기화 방법 설정. Default: "He"

        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.differentiable = True
        self.x = None
        # 4-dim tensor 대응
        self.orgin_x_shape = None

        if initialize == "He":
            self.parameter["weight"] = np.random.randn(input_size, output_size).astype(np.float32) * (np.sqrt(2 / input_size))
            self.parameter["bias"] = np.zeros(output_size).astype(np.float32)    

        elif initialize == "Xavier":
            self.parameter["weight"] = np.random.randn(input_size, output_size).astype(np.float32) * np.sqrt(1/input_size)
            self.parameter["bias"] = np.zeros(output_size).astype(np.float32)

        elif initialize == "None":
            self.parameter["weight"] = 0.01 * np.random.randn(input_size, output_size).astype(np.float32)
            self.parameter["bias"] = np.zeros(output_size).astype(np.float32)

        else:
            raise ValueError("'initialize' must be 'He' or 'Xavier' or 'None'")


        self.dw = np.zeros_like(self.parameter["weight"])
        self.db = np.zeros_like(self.parameter["bias"])


    def __call__(self, arg: np.ndarray):
        """Wrapper function for _forward()

        Parameters
        ----------
        Args:
            arg (numpy.ndarray): 이전 layer의 forward 결과

        Returns:
            numpy.ndarray
        """
        result = self._forward(arg)
        return result

    
    def __repr__(self):
        return "DenseLayer"

    
    def _forward(self, x: np.ndarray)-> np.ndarray:
        """DenseLayer의 feedforward

        Args:
            x (np.ndarray): 이전 layer의 feedforward 결과

        Returns:
            np.ndarray: DenseLayer의 feedforward 결과
        """
        if self.mixed_precision:
            # sequnce 대응
            if x.ndim == 3:
                batch_size, n_timestep, _ = x.shape
                reshaped_x = x.reshape(batch_size * n_timestep, -1)
                self.x = x
                result = np.matmul(reshaped_x, self.parameter["weight"].astype(np.float16)) + self.parameter["bias"].astype(np.float16)
                result = result.reshape(batch_size, n_timestep, -1)

                return result

            # 4-dim tensor 대응
            self.origin_x_shape = x.shape
            x = x.reshape(x.shape[0], -1)
            self.x = x
            result = np.matmul(x, self.parameter["weight"].astype(np.float16)) + self.parameter["bias"].astype(np.float16)

        else:
            # sequnce 대응
            if x.ndim == 3:
                batch_size, n_timestep, _ = x.shape
                reshaped_x = x.reshape(batch_size * n_timestep, -1)
                self.x = x
                result = np.matmul(reshaped_x, self.parameter["weight"]) + self.parameter["bias"]
                result = result.reshape(batch_size, n_timestep, -1)

                return result

            # 4-dim tensor 대응
            self.origin_x_shape = x.shape
            x = x.reshape(x.shape[0], -1)
            self.x = x
            result = np.matmul(x, self.parameter["weight"]) + self.parameter["bias"]

        return result

    
    def _backward(self, input:np.ndarray)-> np.ndarray:
        """DenseLayer의 backpropagation

        Args:
            input (np.ndarray): 이전 layer의 backpropagation 결과

        Returns:
            np.ndarray: DenseLayer의 backpropagation 결과
        """
        if self.mixed_precision:
            if self.x.ndim == 3:
                x = self.x
                batch_size, n_timestep, _ = x.shape
                input = input.reshape(batch_size * n_timestep, -1)
                reshaped_x = x.reshape(batch_size * n_timestep, -1)

                db = np.sum(input, axis=0)
                dw = np.matmul(reshaped_x.T, input)
                dx = np.matmul(input, self.parameter["weight"].T.astype(np.float16))
                dx = dx.reshape(*x.shape)

                self.dw[...] = dw
                self.db[...] = db

                return dx

            else:
                dx = np.matmul(input, self.parameter["weight"].T.astype(np.float16))
                self.dw = np.matmul(self.x.T, input)
                self.db = np.sum(input, axis=0)

                # for 4-dim tensor
                dx = dx.reshape(*self.origin_x_shape)

        else:
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


    def get_gradient(self)->OrderedDict:
        """Layer의 gradient를 return

        Returns:
            OrderedDict: Layer의 gradient
        """
        grad = OrderedDict()
        grad["dw"] = self.dw.astype(np.float32)
        grad["db"] = self.db.astype(np.float32)

        return grad
    

class Embedding(BaseLayer):
    def __init__(self, parameter, mixed_precision = False):
        """
        Initialize EmbeddingLayer

        Parameters
        ----------
        vocab_size (int) : vocab의 size

        hidden_size (int) : embedding될 representation의 hidden size

        initialize (str, optional) : 가중치 초기화 방법 설정. Default: "He"

        """
        super().__init__()
        self.differentiable = True
        self.index = None
        self.parameter["weight"] = parameter["weight"]
        self.dw = np.zeros_like(self.parameter["weight"])
        self.mixed_precision = mixed_precision


    def __call__(self, arg):
        result = self._forward(arg)
        return result

    
    def __repr__(self):
        return "Embedding"

    
    def _forward(self, index):
        if self.mixed_precision:
            self.index = index
            weight = self.parameter["weight"].astype(np.float16)
            result = weight[index]
        
        else:
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
        if self.mixed_precision:
            dw = self.dw
            dw[...] = 0
            if GPU:
                import cupyx
                cupyx.scatter_add(dw, self.index, input)
            else:
                np.add.at(dw, self.index, input)

        else:
            dw = self.dw
            dw[...] = 0
            if GPU:
                import cupyx
                cupyx.scatter_add(dw, self.index, input)
            else:
                np.add.at(dw, self.index, input)
                
        return None


    def _get_gradient(self):
        dw = self.dw.astype(np.float32)

        return dw


class EmbeddingLayer(BaseLayer):
    def __init__(self, vocab_size: int, hidden_size: int, initialize = "He"):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.differentiable = True

        if initialize == "He":
            self.parameter["weight"] = np.random.randn(vocab_size, hidden_size).astype(np.float32) * (np.sqrt(2 / vocab_size))

        elif initialize == "Xavier":
            self.parameter["weight"] = np.random.randn(vocab_size, hidden_size).astype(np.float32) * np.sqrt(1/vocab_size)

        elif initialize == "None":
            self.parameter["weight"] = 0.01 * np.random.randn(vocab_size, hidden_size).astype(np.float32)

        elif isinstance(initialize, int):
            self.parameter["weight"] = (1/initialize) * np.random.randn(vocab_size, hidden_size).astype(np.float32)
        
        else:
            raise ValueError("'initialize' must be 'He' or 'Xavier' or 'None' or integer")

        self.layer = None
        self.dw = np.zeros_like(self.parameter["weight"]).astype(np.float32)


    def __call__(self, arg):
        result = self._forward(arg)
        return result


    def __repr__(self):
        return "EmbeddingLayer"


    def _forward(self, x):
        # Mixed Precision Training
        if self.mixed_precision:
            batch_size, n_timestep = x.shape

            result = np.empty((batch_size, n_timestep, self.hidden_size), dtype="f")
            self.layer = []

            for timestep in range(n_timestep):
                embedding_cell = Embedding(self.parameter, mixed_precision = True)
                result[:, timestep, :] = embedding_cell._forward(x[:, timestep])
                self.layer.append(embedding_cell)

        else:
            batch_size, n_timestep = x.shape
    
            result = np.empty((batch_size, n_timestep, self.hidden_size), dtype="f")
            self.layer = []
            
            for timestep in range(n_timestep):
                embedding_cell = Embedding(self.parameter)
                result[:, timestep, :] = embedding_cell._forward(x[:, timestep])
                self.layer.append(embedding_cell)

        return result

    def _backward(self, dout):
        # Mixed Precision Training
        if self.mixed_precision:
            w = self.parameter["weight"].astype(np.float16)
            batch_size, n_timestep, hidden_size = dout.shape

            dw = np.zeros_like(w).astype(np.float32)
            for timestep in range(n_timestep):
                embedding_cell = self.layer[timestep]
                embedding_cell._backward(dout[:, timestep, :])
                temp_dw = embedding_cell._get_gradient()
                dw += temp_dw

            self.dw[...] = dw

        else:
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

    def get_gradient(self)->OrderedDict:
        """Layer의 gradient를 return

        Returns:
            OrderedDict: Layer의 gradient
        """        
        grad = OrderedDict()
        grad["dw"] = self.dw.astype(np.float32)

        return grad
    
    def _fp16_grad(self):
        self.dw = np.zeros_like(self.parameter["weight"]).astype(np.float16)


class ConvLayer(BaseLayer):
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
        super().__init__()
        self.differentiable = True
        self.input_channel = input_channel
        self.output_channel = output_channel

        if isinstance(kernel_size, int):
            self.kernel_width = kernel_size
            self.kernel_height = kernel_size

        elif isinstance(kernel_size, tuple):
            self.kernel_height, self.kernel_width = kernel_size

        self.stride = stride
        self.padding = padding
        self.fan_in = self.kernel_width * self.kernel_height * input_channel
        self.fan_out = self.kernel_width * self.kernel_height * output_channel

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


    def __repr__(self) :
        return "ConvLayer"


    def __call__(self, arg):
        result = self._forward(arg)
        return result


    def _forward(self, x):
        # Mixed Precision Training
        if self.mixed_precision:
            n_input, n_input_channel, input_height, input_width = x.shape
            out_height = int(1 + (input_height + self.padding * 2 - self.kernel_height) / self.stride)
            out_width = int(1 + (input_width + self.padding * 2 - self.kernel_width) / self.stride)

            col = self.img2col(x)
            col_weight = self.parameter["weight"].reshape(self.output_channel, -1).T.astype(np.float16)
            result = np.matmul(col, col_weight) + self.parameter["bias"].astype(np.float16)
            result = result.reshape(n_input, out_height, out_width, -1).transpose(0, 3, 1, 2)

            self.x = x
            self.col = col
            self.col_weight = col_weight
        
        else:
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
        # Mixed Precision Training
        if self.mixed_precision:
            input = input.transpose(0,2,3,1).reshape(-1, self.output_channel)

            self.dw = np.matmul(self.col.T, input)
            self.dw = self.dw.transpose(1,0).reshape(self.output_channel, self.input_channel, self.kernel_height, self.kernel_width)
            self.db = np.sum(input, axis=0)

            dcol = np.matmul(input, self.col_weight.T)
            result = self.col2img(dcol, self.x.shape)
            
        else:
            input = input.transpose(0,2,3,1).reshape(-1, self.output_channel)

            self.dw = np.matmul(self.col.T, input)
            self.dw = self.dw.transpose(1,0).reshape(self.output_channel, self.input_channel, self.kernel_height, self.kernel_width)
            self.db = np.sum(input, axis=0)

            dcol = np.matmul(input, self.col_weight.T)
            result = self.col2img(dcol, self.x.shape)

        return result


    def get_gradient(self)->OrderedDict:
        """Layer의 gradient를 return

        Returns:
            OrderedDict: Layer의 gradient
        """        
        grad = OrderedDict()
        grad["dw"] = self.dw.astype(np.float32)
        grad["db"] = self.db.astype(np.float32)

        return grad
    
    
    def _fp16_grad(self):
        pass
    

    def img2col(self, input_data):
        # Mixed Precision Training
        if self.mixed_precision:
            n_input, n_input_channel, input_height, input_width = input_data.shape
            out_height = int((input_height + self.padding * 2 - self.kernel_height) // self.stride + 1)
            out_width = int((input_width + self.padding * 2 -self.kernel_width) // self.stride + 1)

            img = np.pad(input_data, [(0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)], 'constant')
            col = np.zeros((n_input, n_input_channel, self.kernel_height, self.kernel_width, out_height, out_width)).astype(np.float32)

            for y in range(self.kernel_height):
                y_max = y + self.stride * out_height
                for x in range(self.kernel_width):
                    x_max = x + self.stride * out_width
                    col[:, :, y, x, :, :] = img[:, :, y:y_max:self.stride, x:x_max:self.stride]

            col = col.transpose(0, 4, 5, 1, 2, 3).reshape(n_input * out_height * out_width, -1)
            
        else:
            n_input, n_input_channel, input_height, input_width = input_data.shape
            out_height = int((input_height + self.padding * 2 - self.kernel_height) // self.stride + 1)
            out_width = int((input_width + self.padding * 2 -self.kernel_width) // self.stride + 1)

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
        # Mixed Precision Training
        if self.mixed_precision:
            n_input, n_input_channel, input_height, input_width = input_shape
            out_height = int((input_height + 2 * self.padding - self.kernel_height) // self.stride + 1)
            out_width = int((input_width + 2 * self.padding - self.kernel_width) // self.stride + 1)
            col = col.reshape(n_input, out_height, out_width, n_input_channel, self.kernel_height, self.kernel_width).transpose(0, 3, 4, 5, 1, 2)

            img = np.zeros((n_input, n_input_channel, input_height + 2 * self.padding + self.stride - 1, input_width + 2 * self.padding + self.stride - 1)).astype(np.float32)
            for y in range(self.kernel_height):
                y_max = y + self.stride * out_height
                for x in range(self.kernel_width):
                    x_max = x + self.stride * out_width
                    img[:, :, y:y_max:self.stride, x:x_max:self.stride] += col[:, :, y, x, :, :]
                    
        else:  
            n_input, n_input_channel, input_height, input_width = input_shape
            out_height = int((input_height + 2 * self.padding - self.kernel_height) // self.stride + 1)
            out_width = int((input_width + 2 * self.padding - self.kernel_width) // self.stride + 1)
            col = col.reshape(n_input, out_height, out_width, n_input_channel, self.kernel_height, self.kernel_width).transpose(0, 3, 4, 5, 1, 2)

            img = np.zeros((n_input, n_input_channel, input_height + 2 * self.padding + self.stride - 1, input_width + 2 * self.padding + self.stride - 1)).astype(np.float32)
            for y in range(self.kernel_height):
                y_max = y + self.stride * out_height
                for x in range(self.kernel_width):
                    x_max = x + self.stride * out_width
                    img[:, :, y:y_max:self.stride, x:x_max:self.stride] += col[:, :, y, x, :, :]

        return img[:, :, self.padding:input_height + self.padding, self.padding:input_width + self.padding]


class GlobalAveragePoolingLayer(BaseLayer):
    def __init__(self):
        super().__init__()
        self.x = None
        
    def __repr__(self):
        return "GlobalAveragePoolingLayer"
    
    
    def __call__(self, arg):
        result = self._forward(arg)
        return result    


    def _forward(self, x):
        n_input, n_input_channel, input_height, input_width = x.shape
        self.x = x
        
        reshaped_x = x.reshape(n_input, n_input_channel, -1)
        
        # result = (n_input, n_input_channel)
        result = reshaped_x.mean(axis=2)

        return result

    
    def _backward(self, input):
        # input = (n_input, n_input_channel)
        n_input, n_input_channel, input_height, input_width = self.x.shape
        temp_result = input * (1/input_height * input_width)
    
        temp_reshaped = temp_result.reshape(n_input, n_input_channel, 1, 1)
        temp_reshaped = np.repeat(temp_reshaped, input_width, axis=3)
        result = np.repeat(temp_reshaped, input_height, axis=2)
        
        return result  
    

class MaxPoolingLayer(BaseLayer):
    def __init__(self, kernel_size, stride = 1, padding = 0):
        super().__init__()
        
        if isinstance(kernel_size, int):
            self.kernel_width = kernel_size
            self.kernel_height = kernel_size

        elif isinstance(kernel_size, tuple):
            self.kernel_height, self.kernel_width = kernel_size

        self.stride = stride
        self.padding = padding

        self.x = None
        self.mask = None

    def __repr__(self):
        return "MaxPoolingLayer"


    def __call__(self, arg):
        result = self._forward(arg)
        return result


    def _forward(self, x):
        # Mixed Precision Training
        if self.mixed_precision:
            n_input, n_input_channel, input_height, input_width = x.shape

            out_height = int(1 + (input_height + self.padding * 2 - self.kernel_height) / self.stride)
            out_width = int(1 + (input_width + self.padding * 2 - self.kernel_width) / self.stride)

            col = self.img2col(x)
            col = col.reshape(-1, self.kernel_height * self.kernel_width)

            self.x = x
            self.mask = np.argmax(col, axis=1) # (-1, self.kernel_height*kernel_width)
            result = np.max(col, axis=1)
            result = result.reshape(n_input, out_height, out_width, n_input_channel).transpose(0, 3, 1, 2)
            
        else:
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
        # Mixed Precision Training
        if self.mixed_precision:
            input = input.transpose(0, 2, 3, 1) # (n_input, n_input_channel, input_height, input_width) -> (n_input, out_height, out_width, n_input_channel)
            kernel_size = self.kernel_height * self.kernel_width
            dmax = np.zeros((input.size, kernel_size)).astype(np.float32) # (n_input*n_input_channel*input_height*input_width, self.kernel_height*kernel_width)
            dmax[np.arange(self.mask.size), self.mask.flatten()] = input.flatten()
            dmax = dmax.reshape(input.shape + (kernel_size,))

            dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
            result = self.col2img(dcol, self.x.shape)
            
        else:
            input = input.transpose(0, 2, 3, 1) # (n_input, n_input_channel, input_height, input_width) -> (n_input, out_height, out_width, n_input_channel)
            kernel_size = self.kernel_height * self.kernel_width
            dmax = np.zeros((input.size, kernel_size)).astype(np.float32) # (n_input*n_input_channel*input_height*input_width, self.kernel_height*kernel_width)
            dmax[np.arange(self.mask.size), self.mask.flatten()] = input.flatten()
            dmax = dmax.reshape(input.shape + (kernel_size,))

            dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
            result = self.col2img(dcol, self.x.shape)

        return result  

    
    def img2col(self, input_data):
        # Mixed Precision Training
        if self.mixed_precision:
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
            
        else:
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
        # Mixed Precision Training
        if self.mixed_precision:
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
                    
        else:
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


class RNNCell(BaseLayer):
    def __init__(self, parameter, mixed_precision = False):
        super().__init__()
        self.differentiable = True  
        self.parameter["weight_x"] = parameter["weight_x"]
        self.parameter["weight_h"] = parameter["weight_h"]
        self.parameter["bias"] = parameter["bias"] 
        self.dx = None
        self.dwx = None
        self.dwh = None
        self.db = None
        self.cache = None
        self.mixed_precision = mixed_precision


    def __repr__(self):
        return "RNNCell"


    def __call__(self, *arg):
        result = self._forward(*arg)
        return result

    
    def _forward(self, x, h_t_prev):
        # Mixed Precision Training
        if self.mixed_precision:
            # (batch_size, hidden_size) x (hidden_size, hidden_size) + (batch_size, input_dim) x (input_dim, hidden_size)
            # => (batch_size, hidden_size)
            temp_t = np.matmul(h_t_prev, self.parameter["weight_h"].astype(np.float16)) + np.matmul(x, self.parameter["weight_x"].astype(np.float16)) + self.parameter["bias"].astype(np.float16) 
            result_t = np.tanh(temp_t)
            # self.cache에 현재 timestep에서의 input, 이전 timestep에서의 hidden state, 현재 timestep에서의 output 저장
            self.cache = x, h_t_prev, result_t
            
        else:
            # (batch_size, hidden_size) x (hidden_size, hidden_size) + (batch_size, input_dim) x (input_dim, hidden_size)
            # => (batch_size, hidden_size)
            temp_t = np.matmul(h_t_prev, self.parameter["weight_h"]) + np.matmul(x, self.parameter["weight_x"]) + self.parameter["bias"] 
            result_t = np.tanh(temp_t)
            # self.cache에 현재 timestep에서의 input, 이전 timestep에서의 hidden state, 현재 timestep에서의 output 저장
            self.cache = x, h_t_prev, result_t

        # (batch_size, hidden_size)
        return result_t


    def _backward(self, input):
        # Mixed Precision Training
        if self.mixed_precision:
            # self.cache에 저장된 현재 timestep에서의 input, 이전 timestep에서의 hidden state, 현재 timestep에서의 output 불러오기
            x, h_t_prev, result_t = self.cache
            # dtanh = 1 - tanh(x)^2
            dtanh = input * (1 - result_t ** 2)
            self.db = np.sum(dtanh, axis=0)
            self.dwh = np.matmul(h_t_prev.T, dtanh)
            self.dwx = np.matmul(x.T, dtanh)
            h_result = np.matmul(dtanh, self.parameter["weight_h"].T.astype(np.float16))
            x_result = np.matmul(dtanh, self.parameter["weight_x"].T.astype(np.float16))
            self.dx = x_result
            
        else:
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


class RNNLayer(BaseLayer):
    def __init__(self, input_size, hidden_size, n_layers = 1, bidirectional = True, stateful = True, initialize = "He"):
        super().__init__()
        self.differentiable = True
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
        self.stateful = stateful
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
        # Mixed Precision Training
        if self.mixed_precision:
            wx, wh, b = self.parameter["weight_x"].astype(np.float16), self.parameter["weight_h"].astype(np.float16), self.parameter["bias"].astype(np.float16)
            batch_size, n_timestep, input_dim = x.shape
            input_dim, hidden_size = wx.shape
            
            hidden_state = np.empty((batch_size, n_timestep, hidden_size) ,dtype="f")
            self.layer = []

            if not self.stateful or self.h is None:
                self.h = np.zeros((batch_size, hidden_size)).astype(np.float32)
            
            for timestep in range(n_timestep):
                rnn_cell = RNNCell(self.parameter, mixed_precision=True)
                # self.h : (batch_size, 1, hidden_size)
                self.h = rnn_cell(x[:, timestep, :], self.h)
                hidden_state[:, timestep, :] = self.h
                self.layer.append(rnn_cell)
                
        else:
            wx, wh, b = self.parameter["weight_x"], self.parameter["weight_h"], self.parameter["bias"]
            batch_size, n_timestep, input_dim = x.shape
            input_dim, hidden_size = wx.shape
            
            hidden_state = np.empty((batch_size, n_timestep, hidden_size) ,dtype="f")
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
        # Mixed Precision Training
        if self.mixed_precision:
            wx, wh, b = self.parameter["weight_x"].astype(np.float16), self.parameter["weight_h"].astype(np.float16), self.parameter["bias"].astype(np.float16)
            batch_size, n_timestep, hidden_size  = dh.shape
            input_dim, hidden_size = wx.shape
            
            dx = np.empty((batch_size, n_timestep, input_dim) ,dtype="f")
            
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
            
        else:
            wx, wh, b = self.parameter["weight_x"], self.parameter["weight_h"], self.parameter["bias"]
            batch_size, n_timestep, hidden_size  = dh.shape
            input_dim, hidden_size = wx.shape
            
            dx = np.empty((batch_size, n_timestep, input_dim) ,dtype="f")
            
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


    def get_gradient(self)-> OrderedDict:
        """Layer의 gradient를 return

        Returns:
            OrderedDict: Layer의 gradient
        """
        grad = OrderedDict()
        grad["dwx"] = self.dwx.astype(np.float32)
        grad["dwh"] = self.dwh.astype(np.float32)
        grad["db"] = self.db.astype(np.float32)

        return grad
    
    
    def load_state(self, h):
        self.h = h


    def reset_state(self):
        self.h = None
    
    
    def _fp16_grad(self):
        pass
        
        
class LSTMCell(BaseLayer):
    def __init__(self, parameter, mixed_precision = False):
        super().__init__()
        self.differentiable = True
        self.parameter["weight_x"] = parameter["weight_x"]
        self.parameter["weight_h"] = parameter["weight_h"]
        self.parameter["bias"] = parameter["bias"]
        self.cache = None
        self.dx = None
        self.dwx = None
        self.dwh = None
        self.db = None
        self.mixed_precision = mixed_precision
        
        
    def __call__(self, *args):
        result = self._forward(*args)
        return result
        
        
    def _forward(self, x, h_t_prev, c_t_prev):
        # Mixed Precision Training
        if self.mixed_precision:
            wx, wh, b = self.parameter["weight_x"].astype(np.float16), self.parameter["weight_h"].astype(np.float16), self.parameter["bias"].astype(np.float16)
            batch_size, hidden_size = h_t_prev.shape
            
            a = np.matmul(x, wx) + np.matmul(h_t_prev, wh) + b
            f_temp = a[:, :hidden_size]
            g_temp = a[:, hidden_size:hidden_size*2]
            i_temp = a[:, hidden_size*2:hidden_size*3]
            o_temp = a[:, hidden_size*3:]
            
            f_result = sigmoid(f_temp).astype(np.float32)
            g_result = np.tanh(g_temp).astype(np.float32)
            i_result = sigmoid(i_temp).astype(np.float32)
            o_result = sigmoid(o_temp).astype(np.float32)
            
            c_t = f_result * c_t_prev + g_result * i_result
            h_t = o_result * np.tanh(c_t)
            
            self.cache = (x, h_t_prev, c_t_prev, f_result, g_result, i_result, o_result, c_t)
        else:
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
        # Mixed Precision Training
        if self.mixed_precision:
            
            wx, wh, b = self.parameter["weight_x"].astype(np.float16), self.parameter["weight_h"].astype(np.float16), self.parameter["bias"].astype(np.float16)
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
            
        else:
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
        dx = self.dx
        dwx = self.dwx
        dwh = self.dwh
        db = self.db

        return dx, dwx, dwh, db
    
    
class LSTMLayer(BaseLayer):
    def __init__(self, input_size, hidden_size, n_layers = 1, bidirectional = True, stateful = True, initialize = "He"):
        super().__init__()
        self.differentiable = True
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
        self.stateful = stateful
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
        # Mixed Precision Training
        if self.mixed_precision:
            wx, wh, b = self.parameter["weight_x"].astype(np.float16), self.parameter["weight_h"].astype(np.float16), self.parameter["bias"].astype(np.float16)
            batch_size, n_timestep, input_dim = x.shape

            hidden_size = wh.shape[0]
            
            self.layer = []
            hidden_state = np.empty((batch_size, n_timestep, hidden_size), dtype="f")
            
            if not self.stateful or self.h is None:
                self.h = np.zeros((batch_size, hidden_size)).astype(np.float32)
            if not self.stateful or self.c is None:
                self.c = np.zeros((batch_size, hidden_size)).astype(np.float32)
            
            for timestep in range(n_timestep):
                lstm_cell = LSTMCell(self.parameter, mixed_precision=True)
                self.h, self.c = lstm_cell(x[:, timestep, :], self.h, self.c)
                hidden_state[:, timestep, :] = self.h
                
                self.layer.append(lstm_cell)
                    
        else:
            wx, wh, b = self.parameter["weight_x"], self.parameter["weight_h"], self.parameter["bias"]
            batch_size, n_timestep, input_dim = x.shape
            hidden_size = wh.shape[0]
            
            self.layer = []
            hidden_state = np.empty((batch_size, n_timestep, hidden_size), dtype="f")
            
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
        # Mixed Precision Training
        if self.mixed_precision:
            wx, wh, b = self.parameter["weight_x"].astype(np.float16), self.parameter["weight_h"].astype(np.float16), self.parameter["bias"].astype(np.float16)
            batch_size, n_timestep, hidden_size = dh.shape
            input_dim = wx.shape[0]
            
            dx = np.empty((batch_size, n_timestep, input_dim), dtype="f")
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
            
            self.dwx = dwx
            self.dwh = dwh
            self.db = db
            self.dh = dh_t
        
        else:
            wx, wh, b = self.parameter["weight_x"], self.parameter["weight_h"], self.parameter["bias"]
            batch_size, n_timestep, hidden_size = dh.shape
            input_dim = wx.shape[0]
            
            dx = np.empty((batch_size, n_timestep, input_dim), dtype="f")
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
            
            self.dwx = dwx
            self.dwh = dwh
            self.db = db
            self.dh = dh_t
            
        return dx
    
    
    def get_gradient(self):
        grad = OrderedDict()
        grad["dwx"] = self.dwx.astype(np.float32)
        grad["dwh"] = self.dwh.astype(np.float32)
        grad["db"] = self.db.astype(np.float32)

        return grad
    
    
    def load_state(self, h, c=None):
        self.h, self.c = h, c
    
    
    def reset_state(self):
        self.h, self.c = None, None 
        

    def _fp16_grad(self):
        pass
    

class BatchNorm1D(BaseLayer):
    def __init__(self, num_features, eps=1e-05, momentum=0.9):
        super().__init__()
        self.changeability = True
        self.differentiable = True
        self.train_fig=True
        self.num_features = num_features

        self.parameter["gamma"] = np.ones(num_features).astype(np.float32)
        self.parameter["beta"] = np.zeros(num_features).astype(np.float32)

        self.eps = eps
        self.momentum = momentum
        
        self.input_reshaped = None
        self.input_shape = None
        self.mean = None
        self.var = None
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None
        
        
    def __call__(self, *args):
        result = self._forward(*args)
        return result


    def __repr__(self):
        return "BatchNormLayer"
    

    def _forward(self, x):
        self.input_shape = x.shape
        if self.input_reshaped == None:
            self._check_input(x)
        
        if x.ndim != 2:
            batch_size, n_input_channel, input_length, = x.shape
            x = x.reshape(batch_size, -1)

        out = self.__forward(x)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, x):
        # Mixed Precision Training
        if self.mixed_precision:
            if self.mean is None:
                batch_size, input_dim = x.shape
                self.mean = np.zeros(input_dim).astype(np.float32)
                self.var = np.zeros(input_dim).astype(np.float32)
                            
            if self.train_fig:
                mu = x.mean(axis=0)
                xc = x - mu
                var = np.mean(xc**2, axis=0)
                std = np.sqrt(var + self.eps)
                xn = xc / std
                
                self.batch_size = x.shape[0]
                self.xc = xc
                self.xn = xn
                self.std = std
                self.mean = self.momentum * self.mean + (1-self.momentum) * mu
                self.var = self.momentum * self.var + (1-self.momentum) * var            
            else:
                xc = x - self.mean
                xn = xc / ((np.sqrt(self.var + self.eps)))
                
            out = self.parameter["gamma"].astype(np.float16) * xn + self.parameter["beta"].astype(np.float16)

        else:
            if self.mean is None:
                batch_size, input_dim = x.shape
                self.mean = np.zeros(input_dim).astype(np.float32)
                self.var = np.zeros(input_dim).astype(np.float32)
                            
            if self.train_fig:
                mu = x.mean(axis=0)
                xc = x - mu
                var = np.mean(xc**2, axis=0)
                std = np.sqrt(var + self.eps)
                xn = xc / std
                
                self.batch_size = x.shape[0]
                self.xc = xc
                self.xn = xn
                self.std = std
                self.mean = self.momentum * self.mean + (1-self.momentum) * mu
                self.var = self.momentum * self.var + (1-self.momentum) * var            
            else:
                xc = x - self.mean
                xn = xc / ((np.sqrt(self.var + self.eps)))
                
            out = self.parameter["gamma"] * xn + self.parameter["beta"]
        
        return out

    def _backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        # Mixed Precision Training
        if self.mixed_precision:
            dbeta = dout.sum(axis=0)
            dgamma = np.sum(self.xn * dout, axis=0)
            dxn = self.parameter["gamma"].astype(np.float16) * dout
            dxc = dxn / self.std
            dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
            dvar = 0.5 * dstd / self.std
            dxc += (2.0 / self.batch_size) * self.xc * dvar
            dmu = np.sum(dxc, axis=0)
            dx = dxc - dmu / self.batch_size
            
            self.dgamma = dgamma
            self.dbeta = dbeta
            
        else:
            dbeta = dout.sum(axis=0)
            dgamma = np.sum(self.xn * dout, axis=0)
            dxn = self.parameter["gamma"] * dout
            dxc = dxn / self.std
            dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
            dvar = 0.5 * dstd / self.std
            dxc += (2.0 / self.batch_size) * self.xc * dvar
            dmu = np.sum(dxc, axis=0)
            dx = dxc - dmu / self.batch_size
            
            self.dgamma = dgamma
            self.dbeta = dbeta
        
        return dx
    
    def _check_input(self, x):
        if x.ndim != 2:
            batch_size, n_input_channel, input_length = self.input_shape
            self.parameter["gamma"] = np.ones(self.num_features*input_length).astype(np.float32)
            self.parameter["beta"] = np.zeros(self.num_features*input_length).astype(np.float32)
            self.input_reshaped = True
        else:
            self.input_reshaped = False
    
    def get_gradient(self)->OrderedDict:
        """Layer의 gradient를 return

        Returns:
            OrderedDict: Layer의 gradient
        """
        grad = OrderedDict()
        grad["dgamma"] = self.dgamma.astype(np.float32)
        grad["dbeta"] = self.dbeta.astype(np.float32)

        return grad
    
    
    def train_state(self):
        self.train_fig = True
        
    
    def eval_state(self):
        self.train_fig = False
    
    
class BatchNorm2D(BaseLayer):
    def __init__(self, num_features, eps=1e-05, momentum=0.9):
        super().__init__()
        self.changeability = True
        self.differentiable = True
        self.train_fig=True
        self.num_features = num_features
        self.parameter["gamma"] = None
        self.parameter["beta"] = None
        self.eps = eps
        self.momentum = momentum
        
        self.input_reshaped = None
        self.input_shape = None
        self.mean = None
        self.var = None
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None
        
        
    def __call__(self, args):
        result = self._forward(args)
        return result


    def __repr__(self):
        return "BatchNormLayer"
    

    def _forward(self, x):
        self.input_shape = x.shape
        # print(x.shape)
        if self.input_reshaped == None:
            self._initialize_param()

        batch_size, n_input_channel, input_height, input_width = x.shape
        x = x.reshape(batch_size, -1)

        out = self.__forward(x)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, x):
        # Mixed Precision Training
        if self.mixed_precision:
            if self.mean is None:
                batch_size, input_dim = x.shape
                self.mean = np.zeros(input_dim).astype(np.float32)
                self.var = np.zeros(input_dim).astype(np.float32)
                            
            if self.train_fig:
                mu = x.mean(axis=0)
                xc = x - mu
                var = np.mean(xc**2, axis=0)
                std = np.sqrt(var + self.eps)
                xn = xc / std
                
                self.batch_size = x.shape[0]
                self.xc = xc
                self.xn = xn
                self.std = std
                self.mean = self.momentum * self.mean + (1-self.momentum) * mu
                self.var = self.momentum * self.var + (1-self.momentum) * var            
            else:
                xc = x - self.mean
                xn = xc / ((np.sqrt(self.var + self.eps)))
                
            out = self.parameter["gamma"].astype(np.float16) * xn + self.parameter["beta"].astype(np.float16)
            
        else:
            if self.mean is None:
                batch_size, input_dim = x.shape
                self.mean = np.zeros(input_dim).astype(np.float32)
                self.var = np.zeros(input_dim).astype(np.float32)
                            
            if self.train_fig:
                mu = x.mean(axis=0)
                xc = x - mu
                var = np.mean(xc**2, axis=0)
                std = np.sqrt(var + self.eps)
                xn = xc / std
                
                self.batch_size = x.shape[0]
                self.xc = xc
                self.xn = xn
                self.std = std
                self.mean = self.momentum * self.mean + (1-self.momentum) * mu
                self.var = self.momentum * self.var + (1-self.momentum) * var            
            else:
                xc = x - self.mean
                xn = xc / ((np.sqrt(self.var + self.eps)))
                
            out = self.parameter["gamma"] * xn + self.parameter["beta"]
            
        return out


    def _backward(self, dout):
        N, C, H, W = dout.shape
        dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx


    def __backward(self, dout):
        # Mixed Precision Training
        if self.mixed_precision:
            dbeta = dout.sum(axis=0)
            dgamma = np.sum(self.xn * dout, axis=0)
            dxn = self.parameter["gamma"].astype(np.float16) * dout
            dxc = dxn / self.std
            dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
            dvar = 0.5 * dstd / self.std
            dxc += (2.0 / self.batch_size) * self.xc * dvar
            dmu = np.sum(dxc, axis=0)
            dx = dxc - dmu / self.batch_size
            
            self.dgamma = dgamma
            self.dbeta = dbeta
            
        else:
            dbeta = dout.sum(axis=0)
            dgamma = np.sum(self.xn * dout, axis=0)
            dxn = self.parameter["gamma"] * dout
            dxc = dxn / self.std
            dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
            dvar = 0.5 * dstd / self.std
            dxc += (2.0 / self.batch_size) * self.xc * dvar
            dmu = np.sum(dxc, axis=0)
            dx = dxc - dmu / self.batch_size
            
            self.dgamma = dgamma
            self.dbeta = dbeta
        
        return dx
    
    
    def _initialize_param(self):
        batch_size, n_input_channel, input_height, input_width = self.input_shape
        self.parameter["gamma"] = np.ones(self.num_features*input_height*input_width).astype(np.float32)
        self.parameter["beta"] = np.zeros(self.num_features*input_height*input_width).astype(np.float32)
        self.input_reshaped = True
    
    
    def get_gradient(self)->OrderedDict:
        """Layer의 gradient를 return

        Returns:
            OrderedDict: Layer의 gradient
        """
        grad = OrderedDict()
        grad["dgamma"] = self.dgamma.astype(np.float32)
        grad["dbeta"] = self.dbeta.astype(np.float32)

        return grad
    
    
    def train_state(self):
        self.train_fig = True
        
    
    def eval_state(self):
        self.train_fig = False


    def _fp16_grad(self):
        pass
    
    
class LayerNorm(BaseLayer):
    def __init__(self, input_shape, eps=1e-05, momentum=0.9):
        super().__init__()
        self.changeability = False
        self.differentiable = True
        if isinstance(input_shape, int):
            input_shape = (input_shape, )
        self.input_shape = tuple(input_shape)

        self.parameter["gamma"] = np.ones(self.input_shape).astype(np.float32)
        self.parameter["beta"] = np.zeros(self.input_shape).astype(np.float32)

        self.eps = eps
        self.input_shape = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None
        
        
    def __call__(self, *args):
        result = self._forward(*args)
        return result


    def __repr__(self):
        return "LayerNormLayer"
    

    def _forward(self, x):
        # Mixed Precision Training
        if self.mixed_precision:
            dims = tuple([-(i+1) for i in range(len(x.shape[1:]))])
            # mu.shape : (batch_size, 1, 1)
            mu = x.mean(axis=dims, keepdims = True)
            xc = x - mu
            var = np.mean(xc**2, axis=dims, keepdims = True)
            std = np.sqrt(var + self.eps)
            xn = xc / std
            
            self.xc = xc
            self.xn = xn
            self.std = std 
     
            out = self.parameter["gamma"].astype(np.float16) * xn + self.parameter["beta"].astype(np.float16)

        else:
            dims = tuple([-(i+1) for i in range(len(x.shape[1:]))])
            # mu.shape : (batch_size, 1, 1)
            mu = x.mean(axis=dims, keepdims = True)
            xc = x - mu
            var = np.mean(xc**2, axis=dims, keepdims = True)
            std = np.sqrt(var + self.eps)
            xn = xc / std
            
            self.xc = xc
            self.xn = xn
            self.std = std 
                
            out = self.parameter["gamma"] * xn + self.parameter["beta"]
        
        return out


    def _backward(self, dout):
        # Mixed Precision Training
        if self.mixed_precision:
            dbeta = dout.sum(axis=0)
            dbeta = dbeta.sum(axis=0)
            dgamma = np.sum(self.xn * dout, axis=0)
            dgamma = dgamma.sum(axis=0)
            dxn = self.parameter["gamma"].astype(np.float16) * dout
            dxc = dxn / self.std
            dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
            dvar = 0.5 * dstd / self.std
            dxc += (2.0 / self.xc[0].size) * self.xc * dvar
            
            dmu = np.sum(dxc, axis=0)
            dx = dxc - dmu / self.xc[0].size

            self.dgamma = dgamma
            self.dbeta = dbeta
            
        else:
            dbeta = dout.sum(axis=0)
            dgamma = np.sum(self.xn * dout, axis=0)
            dxn = self.parameter["gamma"] * dout
            dxc = dxn / self.std
            dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
            dvar = 0.5 * dstd / self.std
            dxc += (2.0 / self.xc[0].size) * self.xc * dvar
            
            dmu = np.sum(dxc, axis=0)
            dx = dxc - dmu / self.xc[0].size

            self.dgamma = dgamma
            self.dbeta = dbeta
        
        return dx
    
    def get_gradient(self)->OrderedDict:
        """Layer의 gradient를 return

        Returns:
            OrderedDict: Layer의 gradient
        """
        grad = OrderedDict()
        grad["dgamma"] = self.dgamma.astype(np.float32)
        grad["dbeta"] = self.dbeta.astype(np.float32)

        return grad


class Dropout(BaseLayer):
    def __init__(self, dropout_ratio=0.5):
        super().__init__()
        self.changeability = True
        self.dropout_ratio = dropout_ratio
        self.train_fig=True
        self.mask = None
        
        
    def __repr__(self):
        return "Dropout"


    def __call__(self, args):
        result = self._forward(args)
        return result
    
    
    def _forward(self, x):
        # Mixed Precision Training
        if self.train_fig:
            fig = np.random.rand(*x.shape) > self.dropout_ratio
            scale = 1 / (1.0 - self.dropout_ratio)
            self.mask = fig.astype(np.float32) * scale
            return x * self.mask
        else:
            return x
        
            # if self.train_fig:
            #     self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            #     return x * self.mask
            # else:
            #     return x * (1.0 - self.dropout_ratio)


    def _backward(self, dout):
        return dout * self.mask
    
    
    def train_state(self):
        self.train_fig = True
        
    
    def eval_state(self):
        self.train_fig = False
        

class Model(BaseModel):
    def __init__(self, *layers):
        """매개 변수에 layer 객체를 순차적으로 전달하면서 초기화
        """
        super().__init__()
        self.sequence = []
        self.grad = OrderedDict()
        
        self.tying_weight = False
        
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
        result = self.forward(arg)
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
                    
                elif repr(layer) =="BatchNormLayer":
                    shape = layer.num_features
                    string_list.append(f"{i}. {layer_name} : {layer} ({shape}) \n")
                    
                else:
                    shape = layer.parameter["weight"].shape
                    string_list.append(f"{i}. {layer_name} : {layer} {shape} \n")
            else:
                string_list.append(f"{i}. {layer_name} : {layer}\n")

        structure = structure.join(string_list)
        return structure
            
            
    def forward(self, x):
        input = x
        if self.tying_weight:
            self.tying_forward()
        for layer_name in self.sequence:
            layer = self.network[layer_name]
            y = layer(input)
            input = y

        return input
        
    
    def backward(self, loss):
        result = loss._backward()
        for layer_name in reversed(self.sequence):
            layer = self.network[layer_name]
            result = layer._backward(result)
        
        if self.tying_weight:
            self.tying_backward()


    def get_gradient(self):
        grad = OrderedDict()
        for layer_name in self.sequence:
            layer = self.network[layer_name]
            if layer.differentiable:
                grad[layer_name] = layer.get_gradient()
                
        return grad
    

    def add_layer(self, *layers):
        tuple_layers = layers
        for layer in tuple_layers:            
            if repr(layer) not in self.count_dict.keys():
                self.count_dict[repr(layer)] = 1

            self.layers += (layer,)
            self.network[f"{repr(layer)}{self.count_dict[repr(layer)]}"] = layer
            self.sequence.append(f"{repr(layer)}{self.count_dict[repr(layer)]}")
            self.count_dict[repr(layer)] += 1
        
        
    def train_state(self):
        for layer_name in self.sequence:
            layer = self.network[layer_name]
            if layer.changeability:
                layer.train_state()

    
    def eval_state(self):
        for layer_name in self.sequence:
            layer = self.network[layer_name]
            if layer.changeability:
                layer.eval_state()
                
    
    def reset_rnn_state(self):
        for layer_name in self.sequence:
            layer = self.network[layer_name]
            if repr(layer) == "LSTMLayer" or repr(layer) == "RNNLayer":
                layer.reset_state()
                
    
    def weight_tying(self):
        emb_layer_name = self.sequence[0]
        final_dense_name = self.sequence[-1]
        
        emb_layer = self.network[emb_layer_name]
        final_dense_layer = self.network[final_dense_name]
        
        emb_layer.tied = True
        final_dense_layer.tying = True
        self.tying_weight = True
        
    
    def tying_forward(self):
        emb_layer_name = self.sequence[0]
        final_dense_name = self.sequence[-1]
        
        emb_layer = self.network[emb_layer_name]
        final_dense_layer = self.network[final_dense_name]
        
        emb_layer.parameter["weight"] = final_dense_layer.parameter["weight"].T
        
        
    def tying_backward(self):
        emb_layer_name = self.sequence[0]
        final_dense_name = self.sequence[-1]
        
        emb_layer = self.network[emb_layer_name]
        final_dense_layer = self.network[final_dense_name]

        final_dense_layer.dw += emb_layer.dw.T
        
    
    def to_cpu(self):
        for layer_name in self.sequence:
            layer = self.network[layer_name]
            if layer.differentiable:
                layer._to_cpu()
 
    
    def to_gpu(self):
        for layer_name in self.sequence:
            layer = self.network[layer_name]
            if layer.differentiable:
                layer._to_gpu()
            
    
    def save_params(self, fn:str = None):
        """Model의 parameter .pkl 확장자로 저장

        Args:
            fn (str): parameter를 저장할 경로 + 파일 이름 + .pkl, Defaults to None.
        """
        model_param = OrderedDict()
        for layer_name in self.sequence:
            layer = self.network[layer_name]
            if layer.differentiable:
                param = layer._save_params()
                model_param[layer_name] = param
                
        if fn is None:
            fn = self.__class__.__name__ + '.pkl'
            
        with open(fn, 'wb') as f:
            pickle.dump(model_param, f)
            
    
    def load_params(self, fn:str = None):
        """.pkl 확장자로 저장되어있던 Model의 parameter를 가져온다

        Args:
            fn (str): parameter가 저장된 경로 + 파일 이름 + .pkl, Defaults to None.
        """
        with open(fn, 'rb') as f:
            model_param = deepcopy(pickle.load(f))
            
        for layer_name in self.sequence:
            layer = self.network[layer_name]
            if layer.differentiable:
                layer._load_params(model_param[layer_name])
                
                
    def mixed_precision_on(self, grad_init = True):
        for layer_name in self.sequence:
            layer = self.network[layer_name]
            layer._mixed_precision_training()

                
    def mixed_precision_off(self):
        for layer_name in self.sequence:
            layer = self.network[layer_name]
            layer._mixed_precision_training_off()


            

