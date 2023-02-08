import numpy as np

"""
Zadanie E.4 (Piotr Bosowski)

Wykorzystujac algorytmy genetyczne oraz biblioteki wbudowane w Python
(bez SciPy, itp) zaimplementuj rozwiazanie problemu TSP. Wykorzystujac
matplotlib przygotuj animacje.
"""


def criterion(cities_positions, order):
    # square_rooted_sum_of_squares
    def sss(x: np.ndarray):
        return np.sqrt(np.sum(x ** 2))

    # we start from and finish in the Borderland city of (0., 0.) coords
    cities_sorted = cities_positions[order, :]
    intermediate_steps = sss(cities_sorted[1:] - cities_sorted[:-1])
    # start and finish:
    borderlands = sss(cities_sorted[(0, -1), :])
    return intermediate_steps + borderlands


def crossover(ind_1, ind_2):
    unused_cities = np.ones_like(ind_1, dtype=bool)
    random_mask = np.random.randint(low=0, high=2,
                                    size=unused_cities.shape,
                                    dtype=bool)
    output = np.empty_like(unused_cities, dtype=int)
    # need to do sequentially
    for i in range(len(unused_cities)):
        cities_left = unused_cities * random_mask
        if cities_left[i]:
            output[i] = ind_2[i]
        else:
            output[i] = ind_1[i]
        unused_cities[output[i]] = False
    return output


def tsp_demonstration(n_cities=10,
                      ndim=2,
                      epochs=10,
                      n_individuals=10,
                      n_children=5):
    n_parents = n_individuals - n_children
    cities_positions = np.random.uniform(size=(n_cities, ndim))
    population = np.vstack([np.random.choice(n_cities,
                                             size=n_cities,
                                             replace=False)
                            for _ in range(n_individuals)])
    for epoch in range(epochs):
        fitnesses = np.stack([criterion(cities_positions, ident)
                              for ident in population])
        order = np.argsort(fitnesses)
        population[:, :] = population[order, :]  # sorting population according to ranks
        fitnesses[:] = fitnesses[order]
        weights = 1 / fitnesses
        print(f'Best route so far ({epoch} epochs): {fitnesses[0]}')
        for i in range(n_children):
            parents = np.random.choice(
                n_parents,
                size=2,
                replace=False,
                p=weights[:n_parents] / np.sum(weights[:n_parents]))
            population[-i, :] = crossover(population[parents[0]],
                                          population[parents[1]])


if __name__ == '__main__':
    tsp_demonstration()
