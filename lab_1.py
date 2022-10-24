import numpy as np
import matplotlib.pyplot as plt
import scipy


if __name__ == '__main__':
    x = 1E20
    dx = 1E20
    while x != x + dx:
        dx = dx/2

    print(x - (x + dx), dx)
dbg_stp = 5