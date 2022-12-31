import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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


def show_img(img):
    image = Image.fromarray(np.uint8(img))
    image.show()
