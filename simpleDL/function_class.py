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


class Sigmoid(BaseFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return sigmoid(x)
    

