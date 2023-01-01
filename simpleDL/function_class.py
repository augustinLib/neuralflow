from function import *

class BaseFunction():
    def __init__(self):
        pass

    def __call__(self, arg):
        result = self.forward(arg)
        return result

    def forward(self, x):
        pass

    def __repr__(self) -> str:
        return "Function"


class Step(BaseFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return step_function(x)


class Identity(BaseFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return identity(x)


class Sigmoid(BaseFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return sigmoid(x)
    

class Softmax(BaseFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return softmax(x)


class ReLU(BaseFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return relu(x)