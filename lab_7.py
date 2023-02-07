import numpy as np


def mean_pair_distance(n=100000,  # number of trials
                       a=1.0,  # cube size
                       dims=3,  # number of dims
                       verbose=True):
    starts = np.random.uniform(0.0, a, (n, dims))
    stops = np.random.uniform(0.0, a, (n, dims))
    result = (((stops - starts) ** 2).sum(axis=1) ** 0.5).mean()
    if verbose:
        print(f"The mean dist. between 2 points within "
              f"a {dims}-D cube (uniform dist.) is {result:.4f}")
    return result


if __name__ == '__main__':
    for dims in [1, 2, 3, 5, 10, 25, 50, 100, 200, 500, 1000]:
        mean_pair_distance(dims=dims)
