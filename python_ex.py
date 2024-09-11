import math
import numpy as np
import matplotlib.pyplot as plt

def grad(x):
    return 6*x + 2 + 4*np.cos(x)

def cost(x):
    return 3*x**2 + 2*x + 4*np.sin(x)
def GD(x_init, eta):
    x = [x_init]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return (x, it)

(x1, it1) = GD(-1, 0.01)
print(x1[-1], cost(x1[-1]), it1)