import numpy as np


def newtons_method(f_x, gradient, x_0, epsilon = 1e-10, max_iter = 10, minimum_defined_value=None):
    x_n = np.array(x_0)
    for n in range(max_iter):
        f = f_x(x_n)
        error = abs(f)
        if error < epsilon:
            return x_n, error
        df = np.array(gradient(x_n))
        step = -df*(f/(np.sum(df))**2)
        x_n = x_n + step
        if minimum_defined_value and x_n < minimum_defined_value:
            x_n = minimum_defined_value
    print('Exceeded maximum iterations. No solution found.')
    return x_n, error