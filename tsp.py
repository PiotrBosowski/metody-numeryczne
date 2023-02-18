import numpy as np
from matplotlib import pyplot as plt, animation

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


def crossover(ind_1, ind_2, mutation_chance=0.2):
    random_mask = np.random.randint(low=0, high=2,
                                    size=ind_1.shape,
                                    dtype=bool)
    used_cities = np.zeros_like(random_mask, dtype=bool)
    output = np.empty_like(random_mask, dtype=int)
    # need to do sequentially
    for i in range(len(random_mask)):
        if random_mask[i]:
            city_index = ind_2[i] if not used_cities[ind_2[i]] else ind_1[i]
        else:
            city_index = ind_1[i] if not used_cities[ind_1[i]] else ind_2[i]
        if used_cities[city_index]:
            city_index = np.random.choice(np.where(used_cities == 0)[0])
        used_cities[city_index] = True
        output[i] = city_index
    assert (len(np.unique(output)) == len(ind_1))
    if np.random.uniform() < mutation_chance:
        swap_1, swap_2 = np.random.choice(len(output), replace=False, size=2)
        output[[swap_1, swap_2]] = output[[swap_2, swap_1]]
    return output


def tsp_demonstration(n_cities=15,
                      ndim=2,
                      epochs=100,
                      n_individuals=400,
                      n_children=200,
                      mutation_chance=0.2):
    n_parents = n_individuals - n_children
    cities_positions = np.random.uniform(size=(n_cities, ndim))
    population = np.vstack([np.random.choice(n_cities,
                                             size=n_cities,
                                             replace=False)
                            for _ in range(n_individuals)])
    # initialization:
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.scatter([0.0] + [x for x, _ in cities_positions],
               [0.0] + [y for _, y in cities_positions])
    images = []
    for epoch in range(epochs):
        fitnesses = np.stack([criterion(cities_positions, ident)
                              for ident in population])
        order = np.argsort(fitnesses)
        # sorting population according to ranks
        population[:, :] = population[order, :]
        fitnesses[:] = fitnesses[order]
        weights = 1 / fitnesses
        print(f'Best route so far ({epoch} epochs): {fitnesses[0]}')
        for i in range(n_children):
            parents = np.random.choice(
                n_parents,
                size=2,
                replace=False,
                p=weights[:n_parents] / np.sum(weights[:n_parents]))
            population[n_individuals - i - 1, :] = \
                crossover(population[parents[0]],
                          population[parents[1]],
                          mutation_chance=mutation_chance)
        # plot the best path so far:
        path = np.concatenate(
            ([[0.0, 0.0]], cities_positions[population[0]], [[0.0, 0.0]]),
            axis=0)
        title = plt.text(0.5, 1.01, f'Epoch {epoch}/{epochs}, '
                                    f'loss: {fitnesses[0]:.4f}',
                         horizontalalignment='center',
                         verticalalignment='bottom', transform=ax.transAxes)
        elements = [title]
        for i in range(1, len(path)):
            start = path[i - 1]
            stop = path[i]
            elements.append(ax.plot((start[0], stop[0]), (start[1], stop[1]),
                                 marker='o')[0])
        images.append(elements)
    # Generate the animation image and save
    animated_image = animation.ArtistAnimation(fig, images, interval=100)
    animated_image.save('tsp_result.gif', writer='pillow')


if __name__ == '__main__':
    tsp_demonstration()
