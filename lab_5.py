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


# Results (for repeat=10):
# For 2 atoms and Nelder-Mead method: 0.2739 s.
# For 2 atoms and Powell method: 0.0885 s.
# For 2 atoms and CG method: 0.0694 s.
# For 3 atoms and Nelder-Mead method: 0.2628 s.
# For 3 atoms and Powell method: 0.2758 s.
# For 3 atoms and CG method: 0.3170 s.
# For 4 atoms and Nelder-Mead method: 1.0473 s.
# For 4 atoms and Powell method: 1.6091 s.
# For 4 atoms and CG method: 0.6450 s.
# For 5 atoms and Nelder-Mead method: 1.7482 s.
# For 5 atoms and Powell method: 2.8570 s.
# For 5 atoms and CG method: 1.6341 s.
# For 10 atoms and Nelder-Mead method: 11.2851 s.
# For 10 atoms and Powell method: 2.9235 s.
# For 10 atoms and CG method: 22.6074 s.
# For 20 atoms and Nelder-Mead method: 85.1797 s.
# For 20 atoms and Powell method: 23.8161 s.
# For 20 atoms and CG method: 18.4187 s.
