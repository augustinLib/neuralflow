from neuralflow.gpu import *
from collections import OrderedDict


class BaseOptimizer():
    def __init__(self, lr = 0.01):
        self.lr = lr
        self.param_grad_dict = OrderedDict()
        self.param_grad_dict["weight"] = "dw"
        self.param_grad_dict["weight_x"] = "dwx"
        self.param_grad_dict["weight_h"] = "dwh"
        self.param_grad_dict["bias"] = "db"
        self.param_grad_dict["gamma"] = "dgamma"
        self.param_grad_dict["beta"] = "dbeta"
        
        
        
    def __repr__(self):
        return "Optimizer"


class SGDOptimizer(BaseOptimizer):
    def __init__(self, lr = 0.01):
        super().__init__(lr)
        
    
    def update(self, model):
        # update parameter
        for layer_name in model.sequence:
            layer = model.network[layer_name]

            # only update differentiable layer
            # do not update tied layer
            if layer.differentiable == True and layer.tied == False:
                grad = layer.get_gradient()
                param_list = list(layer.parameter.keys())
                
                for param in param_list:
                    model.network[layer_name].parameter[param] -= (self.lr * grad[self.param_grad_dict[param]])


class MomentumOptimizer(BaseOptimizer):
    def __init__(self, lr=0.01, momentum = 0.9):
        super().__init__(lr)
        self.momentum = momentum
        self.v = None

    
    def update(self, model):
        if self.v is None:

            # initialize v
            self.v = OrderedDict()
            for layer_name in model.sequence:
                layer = model.network[layer_name]
                if layer.differentiable == True and layer.tied == False:
                    param_list = list(layer.parameter.keys())
                    self.v[layer_name] = OrderedDict()

                    for param in param_list:
                        self.v[layer_name][param] = np.zeros_like(layer.parameter[param])

        # update parameter
        for layer_name in model.sequence:
            layer = model.network[layer_name]

            # only update differentiable layer
            # do not update tied layer
            if layer.differentiable == True and layer.tied == False:
                param_list = list(layer.parameter.keys())
                grad = layer.get_gradient()

                for param in param_list:
                    self.v[layer_name][param] = self.momentum * self.v[layer_name][param] - (self.lr * grad[self.param_grad_dict[param]])
                    model.network[layer_name].parameter[param] += self.v[layer_name][param]


class AdaGrad(BaseOptimizer):
    def __init__(self, lr=0.01):
        super().__init__(lr)
        self.h = None


    def update(self, model):
        if self.h is None:
            # initialize h
            self.h = OrderedDict()
            for layer_name in model.sequence:
                layer = model.network[layer_name]
                if layer.differentiable == True and layer.tied == False:
                    self.h[layer_name] = OrderedDict()
                    param_list = list(layer.parameter.keys())
    
                    for param in param_list:
                        self.h[layer_name][param] = np.zeros_like(layer.parameter[param])


        # update parameter
        for layer_name in model.sequence:
            layer = model.network[layer_name]
            
            # only update differentiable layer
            # do not update tied layer
            if layer.differentiable == True and layer.tied == False:
                grad = layer.get_gradient()
                param_list = list(layer.parameter.keys())

                for param in param_list:
                    # update h
                    self.h[layer_name][param] = grad[self.param_grad_dict[param]] * grad[self.param_grad_dict[param]]
                    # update parameter
                    model.network[layer_name].parameter[param] -= self.lr * grad[self.param_grad_dict[param]] / (np.sqrt(self.h[layer_name][param]) + 1e-7)


class Adam(BaseOptimizer):
    def __init__(self, lr=0.001, b1 = 0.9, b2 = 0.999, epsilon = 1e-8):
        super().__init__(lr)
        self.b1 = b1
        self.b2 = b2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.iter = 0
    

    def update(self, model):
        if self.m is None:
            
            #initialize m, v
            self.m, self.v = OrderedDict(), OrderedDict()
            for layer_name in model.sequence:
                layer = model.network[layer_name]
                if layer.differentiable == True and layer.tied == False:
                    self.m[layer_name], self.v[layer_name] = OrderedDict(), OrderedDict()
                    param_list = list(layer.parameter.keys())

                    for param in param_list:
                        self.m[layer_name][param] = np.zeros_like(layer.parameter[param])
                        self.v[layer_name][param] = np.zeros_like(layer.parameter[param])
                        
        self.iter += 1

        for layer_name in model.sequence:
            layer = model.network[layer_name]

            # only update differentiable layer
            # do not update tied layer
            if layer.differentiable == True and layer.tied == False:
                
                grad = layer.get_gradient()
                param_list = list(layer.parameter.keys())

                for param in param_list:
                    # update m, v
                    self.m[layer_name][param] = self.b1 * self.m[layer_name][param] + (1-self.b1) * grad[self.param_grad_dict[param]]
                    self.v[layer_name][param] = self.b2 * self.v[layer_name][param] + (1-self.b2) * (grad[self.param_grad_dict[param]] ** 2)

                    # update m_hat, v_hat
                    m_hat = self.m[layer_name][param] / (1 - (self.b1 ** self.iter))
                    v_hat = self.v[layer_name][param] / (1 - (self.b2 ** self.iter))

                    # update parameter
                    model.network[layer_name].parameter[param] -= self.lr / (np.sqrt(v_hat) + self.epsilon) * m_hat
