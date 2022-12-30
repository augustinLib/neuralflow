import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    y = x > 0
    return y.astype(np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def identity(x):
    return x


def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x-c)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x

    return y
    

def show_function(func, is_relu = False):
    x = np.arange(-5.0, 5.0, 0.1)
    y = func(x)
    plt.plot(x, y)
    if is_relu:
        plt.ylim(-0.1, 5.1)
    else:
        plt.ylim(-0.1, 1.1)
    plt.show()



class SimpleThreeLayerDNN():
    def __init__(self, input_size: int = 2, hidden_size: int = 3, output_size: int = 2, random_init = True):
        self.network = {}

        if random_init:
            self.network["W1"] = np.random.random((input_size, hidden_size))
            self.network["B1"] = np.random.random((hidden_size))
            self.network["W2"] = np.random.random((hidden_size, output_size))
            self.network["B2"] = np.random.random((output_size))
            self.network["W3"] = np.random.random((output_size, output_size))
            self.network["B3"] = np.random.random((output_size))
        else:
            self.network["W1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
            self.network["B1"] = np.array([0.1, 0.2, 0.3])
            self.network["W2"] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
            self.network["B2"] = np.array([0.1, 0.2])
            self.network["W3"] = np.array([[0.1, 0.3], [0.2, 0.4]])
            self.network["B3"] = np.array([0.1, 0.2])


    def __call__(self, args):
        result = self.forward(args)
        return result


    
    def forward(self, x):
        a1 = np.dot(x, self.network["W1"]) + self.network["B1"]
        z1 = sigmoid(a1)
        a2 = np.dot(z1, self.network["W2"]) + self.network["B2"]
        z2 = sigmoid(a2)
        a3 = np.dot(z2, self.network["W3"]) + self.network["B3"]
        y = identity(a3)
        
        return y
