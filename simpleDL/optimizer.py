import numpy as np
from collections import OrderedDict


class BaseOptimizer():
    def __init__(self, lr = 0.01):
        self.lr = lr
        
    def __repr__(self) -> str:
        return "Optimizer"


class SGDOptimizer(BaseOptimizer):
    def __init__(self, lr = 0.01):
        super().__init__(lr)
        
    
    def update(self, model):
        for layer_name in model.sequence:
            layer = model.network[layer_name]
            if layer.differentiable:
                model.network[layer_name].parameter["weight"] -= (self.lr * layer.dw)
                model.network[layer_name].parameter["bias"] -= (self.lr * layer.db)


class MomentumOptimizer(BaseOptimizer):
    def __init__(self, lr=0.01, momentum = 0.9):
        super().__init__(lr)
        self.momentum = momentum
        self.v = None

    
    def update(self, model):
        if self.v is None:
            self.v = {}
            for layer_name in model.sequence:
                layer = model.network[layer_name]
                if layer.differentiable:
                    self.v[layer_name] = OrderedDict()
                    self.v[layer_name]["weight"] = np.zeros_like(layer.parameter["weight"])
                    self.v[layer_name]["bias"] = np.zeros_like(layer.parameter["bias"])


        for layer_name in model.sequence:
            layer = model.network[layer_name]
            if layer.differentiable:
                self.v[layer_name]["weight"] = self.momentum * self.v[layer_name]["weight"] - (self.lr * layer.dw)
                self.v[layer_name]["bias"] = self.momentum * self.v[layer_name]["bias"] - (self.lr * layer.db)

                model.network[layer_name].parameter["weight"] += self.v[layer_name]["weight"]
                model.network[layer_name].parameter["bias"] += self.v[layer_name]["bias"]


class AdaGrad(BaseOptimizer):
    def __init__(self, lr=0.01):
        super().__init__(lr)
        self.h = None


    def update(self, model):
        if self.h is None:
            self.h = OrderedDict()
            for layer_name in model.sequence:
                layer = model.network[layer_name]
                if layer.differentiable:
                    self.h[layer_name] = OrderedDict()
                    self.h[layer_name]["weight"] = np.zeros_like(layer.parameter["weight"])
                    self.h[layer_name]["bias"] = np.zeros_like(layer.parameter["bias"])


        for layer_name in model.sequence:
            layer = model.network[layer_name]
            if layer.differentiable:
                # update h
                self.h[layer_name]["weight"] = layer.dw * layer.dw
                self.h[layer_name]["bias"] = layer.db * layer.db
                
                # update parameter
                model.network[layer_name].parameter["weight"] -= self.lr * layer.dw / (np.sqrt(self.h[layer_name]["weight"]) + 1e-7)
                model.network[layer_name].parameter["bias"] -= self.lr * layer.db / (np.sqrt(self.h[layer_name]["bias"]) + 1e-7)


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
            self.m, self.v = OrderedDict(), OrderedDict()
            for layer_name in model.sequence:
                layer = model.network[layer_name]
                if layer.differentiable:
                    self.m[layer_name], self.v[layer_name] = OrderedDict(), OrderedDict()

                    self.m[layer_name]["weight"] = np.zeros_like(layer.parameter["weight"])
                    self.m[layer_name]["bias"] = np.zeros_like(layer.parameter["bias"])

                    self.v[layer_name]["weight"] = np.zeros_like(layer.parameter["weight"])
                    self.v[layer_name]["bias"] = np.zeros_like(layer.parameter["bias"])

        self.iter += 1

        for layer_name in model.sequence:
            layer = model.network[layer_name]
            if layer.differentiable:
                # update m, v
                self.m[layer_name]["weight"] = self.b1 * self.m[layer_name]["weight"] + (1-self.b1) * layer.dw
                self.m[layer_name]["bias"] = self.b1 * self.m[layer_name]["bias"] + (1-self.b1) * layer.db
                self.v[layer_name]["weight"] = self.b2 * self.v[layer_name]["weight"] + (1-self.b2) * (layer.dw ** 2)
                self.v[layer_name]["bias"] = self.b2 * self.v[layer_name]["bias"] + (1-self.b2) * (layer.db ** 2)

                m_hat_weight = self.m[layer_name]["weight"] / (1 - (self.b1 ** self.iter))
                m_hat_bias = self.m[layer_name]["bias"] / (1 - (self.b1 ** self.iter))
                v_hat_weight = self.v[layer_name]["weight"] / (1 - (self.b2 ** self.iter))
                v_hat_bias = self.v[layer_name]["bias"] / (1 - (self.b2 ** self.iter))
                
                # update parameter
                model.network[layer_name].parameter["weight"] -= self.lr / (np.sqrt(v_hat_weight) + self.epsilon) * m_hat_weight
                model.network[layer_name].parameter["bias"] -= self.lr / (np.sqrt(v_hat_bias) + self.epsilon) * m_hat_bias

