import numpy as np
import matplotlib.pyplot as plt
import scipy


float_type = np.float32


if __name__ == '__main__':
    x =  float_type(1E20)
    dx = float_type(1E20)
    iterations = 0
    while x != x + dx:
        iterations += 1
        dx = dx/float_type(2)
        print(dx)
        if type(dx) is float_type:
            dbg_stp = 4

    print(x - (x + dx), dx, iterations)
dbg_stp = 5