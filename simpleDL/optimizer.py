import numpy as np


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
                    self.v[layer_name] = {}
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
            self.h = {}
            for layer_name in model.sequence:
                layer = model.network[layer_name]
                if layer.differentiable:
                    self.h[layer_name] = {}
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