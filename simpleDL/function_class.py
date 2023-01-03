from simpleDL.function import *

class BaseFunction():
    def __init__(self):
        pass

    def __call__(self, arg):
        result = self._forward(arg)
        return result

    def _forward(self, x):
        pass

    def __repr__(self) -> str:
        return "Function"


class Step(BaseFunction):
    def __init__(self):
        super().__init__()

    def _forward(self, x):
        return step_function(x)


class Identity(BaseFunction):
    def __init__(self):
        super().__init__()

    def _forward(self, x):
        return identity(x)


class Sigmoid(BaseFunction):
    def __init__(self):
        self.differentiable = False
        self.out = None
        super().__init__()

    def _forward(self, x):
        out = sigmoid(x)
        self.out = out

        return out


    def _backward(self, input):
        result = input * self.out * (1.0-self.out)

        return result

    

class Softmax(BaseFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return softmax(x)


class ReLU(BaseFunction):
    def __init__(self):
        self.differentiable = False
        self.mask = None
        super().__init__()

    def _forward(self, x):
        self.mask = (x <= 0)
        result = x.copy()
        result[self.mask] = 0
        
        return result

    def _backward(self, input):
        input[self.mask] = 0
        result = input

        return result



class CrossEntropyLoss():
    def __init__(self):
        self.loss = None
        self.pred = None
        self.true = None

    def __repr__(self) -> str:
        return "Function"


    def __call__(self, *arg):
        result = self._forward(*arg)
        return result

    def _forward(self, pred, true):
        self.pred = softmax(pred)
        self.true = true
        self.loss = cross_entropy_error(self.pred, self.true)

        return self.loss

    def _backward(self):
        batch_size = self.true.shape[0]
        if self.true.size == self.pred.size:
            dx = (self.pred - self.true) / batch_size
        else:
            dx = self.pred.copy()
            dx[np.arange(batch_size), self.true] -= 1
            dx = dx / batch_size

        return dx