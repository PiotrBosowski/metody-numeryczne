import numpy as np
from scipy.optimize import differential_evolution


def rosenbrock(x):
    # works for dim >= 2
    # should always give 0.0 for x = [1., 1., 1., ..., 1.]
    return np.sum((1 - x[:-1]) ** 2) + np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2)


def rosenbrock_diff_evol_experiment(dims):
    single_bound = (-10, 10)
    for dim in dims:
        bounds = [single_bound for _ in range(dim)]
        result = differential_evolution(rosenbrock, bounds)
        print('Number of dims:', dim)
        print('Success:', result.success)
        print('Best x:', result.x)
        print('Num. of evaluations:', result.nfev)


if __name__ == '__main__':
    rosenbrock_diff_evol_experiment([2, 4, 8, 20])




