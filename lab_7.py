import numpy as np
import matplotlib.pyplot as plt
import scipy


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


def decay_function(x, lamb):
    return np.exp(-lamb * x)


def radionuclide_decay(N_0=1000, lamb=0.01, delta_t=1.0, verbose=True,
                       compare_with_theoretical=False):
    history = [1.0]
    threshold = lamb * delta_t
    elems = np.ones((N_0,), dtype=bool)
    epoch = 0
    while np.any(elems):
        epoch += 1
        probs = np.random.uniform(size=(N_0,))
        alive = probs > threshold
        elems *= alive
        alive_rate = elems.mean()
        history.append(alive_rate)
        if verbose:
            print(f"Epoch {epoch}: {alive_rate}")
    if verbose:
        x_ticks = np.arange(0, len(history) * delta_t)
        plt.plot(x_ticks, history)
        if compare_with_theoretical:
            history_theo = decay_function(x_ticks, lamb)
            plt.plot(x_ticks, history_theo)
            analysis = scipy.optimize.curve_fit(decay_function, x_ticks, history)
            print(analysis)
        plt.show()
    return history


if __name__ == '__main__':
    # for dims in [1, 2, 3, 5, 10, 25, 50, 100, 200, 500, 1000]:
    #     mean_pair_distance(dims=dims)
    for N_0 in [5, 25, 125, 625, 3125]:
        radionuclide_decay(N_0=N_0, compare_with_theoretical=True)
