import numpy as np
from matplotlib import pyplot as plt


class LinearApproximator:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self, x):
        for ind, item in enumerate(x):
            if x < item:
                break




def get_derivative_both(x: np.ndarray, y: np.ndarray):
    """
    Assuming x is equidistant.
    :param step_span: indicates how many consecutive values should be
        used to calaulate a local slope
    """
    y_start = y[:-2]
    y_stop = y[2:]
    y_diff = (y_stop - y_start) / (x[2] - x[0])
    x_clip = x[1:-1]
    return x_clip, y_diff


def get_derivative_right(x: np.ndarray, y: np.ndarray, ):
    """
    Assuming x is equidistant.
    :param step_span: indicates how many consecutive values should be
        used to calaulate a local slope
    """
    y_start = y[:-1]
    y_stop = y[1:]
    y_diff = (y_stop - y_start) / (x[1] - x[0])
    x_clip = x[:-1]
    return x_clip, y_diff


if __name__ == '__main__':
    delta_x = .01
    x = np.arange(0., 5., delta_x)
    A = 10.
    beta = .5
    omega = 3.
    y = A * np.exp(-beta * x) * np.cos(omega * x)


    x_clip_both, y_der_both = get_derivative_both(x, y)
    x_clip_right, y_der_right = get_derivative_right(x, y)

    plt.scatter(x[1:-1], y[1:-1], marker='.', label='y')
    plt.scatter(x[1:-1], y_der_both, marker='.', label='y\'_both')
    plt.scatter(x[1:-1], y_der_right[1:], marker='.', label='y\'_right')
    plt.legend()
    plt.show()

    dbg_stp = 5
