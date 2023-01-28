import pickle
from copy import deepcopy
from collections import OrderedDict
from neuralflow.function import *
from neuralflow.function_class import *
from neuralflow.gpu import *    
from neuralflow.utils import *
from neuralflow.model import *


class BaseModule():
    def __init__(self):
        self.parameter = OrderedDict()
        self.differentiable = False
        self.changeability = False
        self.mixed_precision = False
        self.tying = False
        self.tied = False
    
    def __repr__(self):
        return "Module"
    
    
    def _save_params(self):
        """Layer의 parameter 반환

        Returns:
            OrderedDict
        """
        return deepcopy(self.parameter)
    
    
    def _load_params(self, params):
        self.parameter = deepcopy(params)
        

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
    

class ResNet34Module(BaseModule):
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding, residual_connect = True, initialize: str = "He"):
        super().__init__()
        self.conv1 = ConvLayer(input_channel, output_channel, kernel_size, stride, padding)
        self.conv2 = ConvLayer(output_channel, output_channel, kernel_size, stride, padding)
        self.batch_norm1 = BatchNorm2D(output_channel)
        self.batch_norm2 = BatchNorm2D(output_channel)
        self.relu1 = ReLU()
        self.relu2 = ReLU()
        self.residual_connect = residual_connect
        self.first_forwarded = False
        
        self.differentiable = True
        self.changeability = True
        self.bn_not_initialized = True

        if isinstance(kernel_size, int):
            self.kernel_width = kernel_size
            self.kernel_height = kernel_size

        elif isinstance(kernel_size, tuple):
            self.kernel_height, self.kernel_width = kernel_size
            
        self.conv1_fan_in = self.kernel_width * self.kernel_height * input_channel
        self.conv1_fan_out = self.kernel_width * self.kernel_height * output_channel
        self.conv2_fan_in = self.kernel_width * self.kernel_height * output_channel
        self.conv2_fan_out = self.kernel_width * self.kernel_height * output_channel
        
        if initialize == "He":
            self.parameter["conv1_weight"] = np.random.randn(output_channel, input_channel, self.kernel_height, self.kernel_width).astype(np.float32) * (np.sqrt(2 / self.conv1_fan_in))
            self.parameter["conv1_bias"] = np.zeros(output_channel).astype(np.float32) 
            self.parameter["conv2_weight"] = np.random.randn(output_channel, output_channel, self.kernel_height, self.kernel_width).astype(np.float32) * (np.sqrt(2 / self.conv2_fan_in))
            self.parameter["conv2_bias"] = np.zeros(output_channel).astype(np.float32) 

        elif initialize == "Xavier":
            self.parameter["conv1_weight"] = np.random.randn(output_channel, input_channel, self.kernel_height, self.kernel_width).astype(np.float32) * (np.sqrt(1 / self.conv1_fan_in))
            self.parameter["conv1_bias"] = np.zeros(output_channel).astype(np.float32) 
            self.parameter["conv2_weight"] = np.random.randn(output_channel, output_channel, self.kernel_height, self.kernel_width).astype(np.float32) * (np.sqrt(1 / self.conv2_fan_in))
            self.parameter["conv2_bias"] = np.zeros(output_channel).astype(np.float32) 

        elif initialize == "None":
            self.parameter["conv1_weight"] = 0.01 * np.random.randn(output_channel, input_channel, self.kernel_height, self.kernel_width).astype(np.float32)
            self.parameter["conv1_bias"] = np.zeros(output_channel).astype(np.float32) 
            self.parameter["conv2_weight"] = 0.01 * np.random.randn(output_channel, output_channel, self.kernel_height, self.kernel_width).astype(np.float32)
            self.parameter["conv2_bias"] = np.zeros(output_channel).astype(np.float32) 

        else:
            raise ValueError("'initialize' must be 'He' or 'Xavier' or 'None'")
        
        self.conv1.parameter["weight"] = self.parameter["conv1_weight"] 
        self.conv1.parameter["bias"] = self.parameter["conv1_bias"] 
        self.conv2.parameter["weight"] = self.parameter["conv2_weight"] 
        self.conv2.parameter["bias"] = self.parameter["conv2_bias"]
        self.parameter["b1_gamma"] 
        self.parameter["b1_beta"] 
        self.parameter["b2_gamma"]
        self.parameter["b2_beta"]
                
        
    def sync_param(self):
        self.parameter["conv1_weight"] = self.conv1.parameter["weight"]
        self.parameter["conv1_bias"] = self.conv1.parameter["bias"]
        self.parameter["conv2_weight"] = self.conv2.parameter["weight"]
        self.parameter["conv2_bias"] = self.conv2.parameter["bias"]
        self.parameter["b1_gamma"] = self.batch_norm1.parameter["gamma"]
        self.parameter["b1_beta"] = self.batch_norm1.parameter["beta"]
        self.parameter["b2_gamma"] = self.batch_norm2.parameter["gamma"]
        self.parameter["b2_beta"] = self.batch_norm2.parameter["beta"]
        
        
    def __repr__(self):
        return "ResidualModule"

    def __call__(self, arg):
        result = self._forward(arg)
        return result
        
        
    def _forward(self, x):
        result = self.conv1(x)
        result = self.batch_norm1(result)
        result = self.relu1(result)
        result = self.conv2(result)
        result = self.batch_norm2(result)
        if self.residual_connect:
            result = np.add(self.relu2(result),x)
        else:
            result = self.relu2(result)
        
        if self.bn_not_initialized:
            self.parameter["b1_gamma"] = self.batch_norm1.parameter["gamma"]
            self.parameter["b1_beta"] = self.batch_norm1.parameter["beta"]
        
            self.parameter["b2_gamma"] = self.batch_norm2.parameter["gamma"]
            self.parameter["b2_beta"] = self.batch_norm2.parameter["beta"]
            self.bn_not_initialized = False
        
        return result
                
    def _backward(self, dout):
        
        result = self.relu2._backward(dout)
        result = self.batch_norm2._backward(result)
        result = self.conv2._backward(result)
        result = self.relu1._backward(result)
        result = self.batch_norm1._backward(result)
        result = self.conv1._backward(result)
        
        return result
    
    def get_gradient(self)->OrderedDict:
        """Layer의 gradient를 return

        Returns:
            OrderedDict: Layer의 gradient
        """        
        grad = OrderedDict()
        grad["c1_dw"] = self.conv1.dw
        grad["c1_db"] = self.conv1.db
        grad["c2_dw"] = self.conv2.dw
        grad["c2_db"] = self.conv2.db
        grad["b1_dgamma"] = self.batch_norm1.dgamma
        grad["b1_dbeta"] = self.batch_norm1.dbeta
        grad["b2_dgamma"] = self.batch_norm2.dgamma
        grad["b2_dbeta"] = self.batch_norm2.dbeta

        return grad
    
    
    def eval_state(self):
        self.batch_norm1.eval_state()
        self.batch_norm2.eval_state()
        
        
    def train_state(self):
        self.batch_norm1.train_state()
        self.batch_norm2.train_state()
        
        
    def sync_param(self):
        self.parameter["conv1_weight"] = self.conv1.parameter["weight"]
        self.parameter["conv1_bias"] = self.conv1.parameter["bias"]
        self.parameter["conv2_weight"] = self.conv2.parameter["weight"]
        self.parameter["conv2_bias"] = self.conv2.parameter["bias"]
        self.parameter["b1_gamma"] = self.batch_norm1.parameter["gamma"]
        self.parameter["b1_beta"] = self.batch_norm1.parameter["beta"]
        self.parameter["b2_gamma"] = self.batch_norm2.parameter["gamma"]
        self.parameter["b2_beta"] = self.batch_norm2.parameter["beta"]
        
        
        
class ResNet34(Model):
    def __init__(self, input_size, num_class):
        super().__init__()
        if isinstance(input_size, int):
            self.input_width = input_size
            self.input_height = input_size

        elif isinstance(input_size, tuple):
            self.input_height, self.kernel_width = input_size
            
            
        self.add_layer(ConvLayer(3, 64, (7,7), stride=2, padding=3),
                       BatchNorm2D(64),
                       ReLU(),
                       MaxPoolingLayer(3, stride = 2, padding = 1)
                       )
        
        self.add_layer(
            ResNet34Module(64, 64, kernel_size=(3,3),stride=1, padding=1),
            ResNet34Module(64, 64, kernel_size=(3,3),stride=1, padding=1),
            ResNet34Module(64, 64, kernel_size=(3,3),stride=1, padding=1)
        )
        
        self.add_layer(
            ResNet34Module(64, 128, kernel_size=(3,3),stride=1, padding=1, residual_connect=False),
            ResNet34Module(128, 128, kernel_size=(3,3),stride=1, padding=1),
            ResNet34Module(128, 128, kernel_size=(3,3),stride=1, padding=1),
            ResNet34Module(128, 128, kernel_size=(3,3),stride=1, padding=1)
        )

        self.add_layer(
            ResNet34Module(128, 256, kernel_size=(3,3),stride=1, padding=1, residual_connect=False),
            ResNet34Module(256, 256, kernel_size=(3,3),stride=1, padding=1),
            ResNet34Module(256, 256, kernel_size=(3,3),stride=1, padding=1),
            ResNet34Module(256, 256, kernel_size=(3,3),stride=1, padding=1),
            ResNet34Module(256, 256, kernel_size=(3,3),stride=1, padding=1),
            ResNet34Module(256, 256, kernel_size=(3,3),stride=1, padding=1)
        )
        
        self.add_layer(
            ResNet34Module(256, 512, kernel_size=(3,3),stride=1, padding=1, residual_connect=False),
            ResNet34Module(512, 512, kernel_size=(3,3),stride=1, padding=1),
            ResNet34Module(512, 512, kernel_size=(3,3),stride=1, padding=1)
        )
        
        self.add_layer(
            GlobalAveragePoolingLayer(),
            DenseLayer(512, 1000),
            BatchNorm1D(1000),
            ReLU(),
            Dropout(0.3),
            DenseLayer(1000, num_class)
)


        

class VGG16(Model):
    def __init__(self, input_size, num_class, input_channel = 3):
        super().__init__()
        if isinstance(input_size, int):
            self.input_width = input_size
            self.input_height = input_size

        elif isinstance(input_size, tuple):
            self.input_height, self.kernel_width = input_size
        

        self.add_layer(ConvLayer(input_channel,64,(3,3), stride=1, padding=1),
                        BatchNorm2D(64),
                        ReLU(),
                        ConvLayer(64,64,(3,3), stride=1, padding=1),
                        BatchNorm2D(64),
                        ReLU(),
                        MaxPoolingLayer((2,2), stride=2))
        
        self.add_layer(ConvLayer(64,128,(3,3), stride=1, padding=1),
                        BatchNorm2D(128),
                        ReLU(),
                        ConvLayer(128,128,(3,3), stride=1, padding=1),
                        BatchNorm2D(128),
                        ReLU(),
                        MaxPoolingLayer((2,2), stride=2))
        
        self.add_layer(ConvLayer(128,256,(3,3), stride=1, padding=1),
                        BatchNorm2D(256),
                        ReLU(),
                        ConvLayer(256,256,(3,3), stride=1, padding=1),
                        BatchNorm2D(256),
                        ReLU(),
                        ConvLayer(256,256,(3,3), stride=1, padding=1),
                        BatchNorm2D(256),
                        ReLU(),
                        MaxPoolingLayer((2,2), stride=2))
                        
        self.add_layer(ConvLayer(256,512,(3,3), stride=1, padding=1),
                        BatchNorm2D(512),
                        ReLU(),
                        ConvLayer(512,512,(3,3), stride=1, padding=1),
                        BatchNorm2D(512),
                        ReLU(),
                        ConvLayer(512,512,(3,3), stride=1, padding=1),
                        BatchNorm2D(512),
                        ReLU(),
                        MaxPoolingLayer((2,2), stride=2))
        
        self.add_layer(ConvLayer(512,512,(3,3), stride=1, padding=1),
                        BatchNorm2D(512),
                        ReLU(),
                        ConvLayer(512,512,(3,3), stride=1, padding=1),
                        BatchNorm2D(512),
                        ReLU(),
                        ConvLayer(512,512,(3,3), stride=1, padding=1),
                        BatchNorm2D(512),
                        ReLU(),
                        MaxPoolingLayer((2,2), stride=2))
        
        self.add_layer(DenseLayer(512*(int(self.input_height//32)*int(self.input_height//32)), 4096),
                       BatchNorm1D(4096),
                       ReLU(),
                       Dropout(0.3),
                       DenseLayer(4096, 4096),
                       BatchNorm1D(4096),
                       ReLU(),
                       Dropout(0.3),
                       DenseLayer(4096, num_class))
        
        

class VGG19(Model):
    def __init__(self, input_size, num_class, input_channel = 3):
        super().__init__()
        if isinstance(input_size, int):
            self.input_width = input_size
            self.input_height = input_size

        elif isinstance(input_size, tuple):
            self.input_height, self.kernel_width = input_size


        self.add_layer(
            ConvLayer(input_channel ,64,(3,3), stride=1, padding=1),
            BatchNorm2D(64),
            ReLU(),
            ConvLayer(64,64,(3,3), stride=1, padding=1),
            BatchNorm2D(64),
            ReLU(),
            MaxPoolingLayer((2,2), stride=2)
            )
        
        self.add_layer(ConvLayer(64,128,(3,3), stride=1, padding=1),
                        BatchNorm2D(128),
                        ReLU(),
                        ConvLayer(128,128,(3,3), stride=1, padding=1),
                        BatchNorm2D(128),
                        ReLU(),
                        MaxPoolingLayer((2,2), stride=2))
        
        self.add_layer(ConvLayer(128,256,(3,3), stride=1, padding=1),
                        BatchNorm2D(256),
                        ReLU(),
                        ConvLayer(256,256,(3,3), stride=1, padding=1),
                        BatchNorm2D(256),
                        ReLU(),
                        ConvLayer(256,256,(3,3), stride=1, padding=1),
                        BatchNorm2D(256),
                        ReLU(),
                        ConvLayer(256,256,(3,3), stride=1, padding=1),
                        BatchNorm2D(256),
                        ReLU(),
                        MaxPoolingLayer((2,2), stride=2))
                        
        self.add_layer(ConvLayer(256,512,(3,3), stride=1, padding=1),
                        BatchNorm2D(512),
                        ReLU(),
                        ConvLayer(512,512,(3,3), stride=1, padding=1),
                        BatchNorm2D(512),
                        ReLU(),
                        ConvLayer(512,512,(3,3), stride=1, padding=1),
                        BatchNorm2D(512),
                        ReLU(),
                        ConvLayer(512,512,(3,3), stride=1, padding=1),
                        BatchNorm2D(512),
                        ReLU(),
                        MaxPoolingLayer((2,2), stride=2))
        
        self.add_layer(ConvLayer(512,512,(3,3), stride=1, padding=1),
                        BatchNorm2D(512),
                        ReLU(),
                        ConvLayer(512,512,(3,3), stride=1, padding=1),
                        BatchNorm2D(512),
                        ReLU(),
                        ConvLayer(512,512,(3,3), stride=1, padding=1),
                        BatchNorm2D(512),
                        ReLU(),
                        ConvLayer(512,512,(3,3), stride=1, padding=1),
                        BatchNorm2D(512),
                        ReLU(),
                        MaxPoolingLayer((2,2), stride=2))
        
        self.add_layer(DenseLayer(512*(int(self.input_height//32)*int(self.input_height//32)), 4096),
                       BatchNorm1D(4096),
                       ReLU(),
                       Dropout(0.3),
                       DenseLayer(4096, 4096),
                       BatchNorm1D(4096),
                       ReLU(),
                       Dropout(0.3),
                       DenseLayer(4096, num_class))
                       

