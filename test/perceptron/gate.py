import numpy as np

def AND(x1, x2):
    input = np.array([x1, x2])
    weight = np.array([0.5, 0.5])
    bias = -0.7

    temp = np.sum(input*weight) + bias
    if temp <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    input = np.array([x1, x2])
    weight = np.array([-0.5, -0.5])
    bias = 0.7

    temp = np.sum(input*weight) + bias
    if temp <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    input = np.array([x1, x2])
    weight = np.array([0.5, 0.5])
    bias = -0.2

    temp = np.sum(input*weight) + bias
    if temp <= 0:
        return 0
    else:
        return 1



def print_result(gate_func: "function", inputs: list):
    print(f"{gate_func.__name__} gate result")
    print(f"x1 | x2  |  y" )
    for x1, x2 in inputs:
        print(x1, " | ", x2, " | ", gate_func(x1, x2))





