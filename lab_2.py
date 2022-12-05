from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import lagrange, splrep, splev


class LinearInterpolation:
    def __init__(self, train_X, train_y):
        self.train_X = train_X
        self.train_y = train_y
        self.domain = train_X[0], train_X[-1]

    def __call__(self, val):
        bot, top = self._find_closest_indices_pair(val)
        return self.train_y[bot] + (val - self.train_X[bot]) /\
            (self.train_X[top] - self.train_X[bot]) \
            * (self.train_y[top] - self.train_y[bot])

    def _find_closest_indices_pair(self, val):
        if val < self.domain[0] or val > self.domain[1]:
            raise RuntimeError
        for i, x in enumerate(self.train_X):
            if val < x:
                break
        return i - 1, i


if __name__ == '__main__':
    x = [1, 2, 3, 4, 5, 10]
    y = [0, 1, 2, 0, 2, 5]
    li = LinearInterpolation(x, y)
    print(li(7.5))
    plt.scatter(x, y)
    plt.show()

    x_inter = np.linspace(1, 10, 100)
    y_inter = [li(x_curr) for x_curr in x_inter]
    plt.scatter(x_inter, y_inter)
    plt.show()

    tck = splrep(x, y)
    y_spl = splev(x_inter, tck)
    plt.scatter(x_inter, y_spl)
    plt.show()

    poly = lagrange(x, y)
    y_poly = np.polynomial.Polynomial(poly.coef[::-1])(x_inter)
    plt.scatter(x_inter, y_poly)
    plt.show()
    dbg_stp = 5
