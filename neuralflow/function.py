from neuralflow.gpu import *
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
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))
    

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


def sseLoss(pred, true):
    return 0.5 * np.sum((pred-true)**2)


def cross_entropy_error(pred, t):
    if pred.ndim == 1:
        t = t.reshape(1, t.size)
        pred = pred.reshape(1, pred.size)
        
    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == pred.size:
        t = t.argmax(axis=1)
             
    batch_size = pred.shape[0]
    return -np.sum(np.log(pred[np.arange(batch_size), t] + 1e-7)) / batch_size


def numerical_diff(function, x):
    h = 1e-4
    return (function(x+h) - function(x-h)) / (2*h)


def numerical_gradient(function, x):
    h = 1e-4
    gradient = np.zeros_like(x)

    for index in range(x.size):
        temp = x[index]
        x[index] = temp + h
        fxh1 = function(x)

        x[index] = temp - h
        fxh2 = function(x)

        gradient[index] = (fxh1 - fxh2) / (2*h)
        x[index] = temp

    return gradient


def gradient_descent(function, init_x, lr=0.01, num_step=100):
    x = init_x

    for i in range(num_step):
        grad = numerical_gradient(function, x)
        x -= lr*grad

    return x