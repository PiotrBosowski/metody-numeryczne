import time

from scipy.optimize import minimize

from lab_4 import fill_cube, get_total_potential


def benchmark(function, *args, repeat=10, **kwargs):
    start = time.perf_counter()
    for _ in range(repeat):
        function(*args, **kwargs)
    return (time.perf_counter() - start) / repeat


if __name__ == '__main__':
    for n in [2, 3, 4, 5, 10, 20]:
        coords = fill_cube(n)
        for method in ['Nelder-Mead',
                       'Powell',
                       'CG']:
            score = benchmark(minimize, get_total_potential, coords,
                              method=method)
            print(f'For {n} atoms and {method} method: {score:.4f} s.')

    dbg_stp = 5
