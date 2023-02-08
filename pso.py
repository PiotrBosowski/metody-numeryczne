import numpy as np
from matplotlib import pyplot as plt, animation


"""
Zadanie E.5 (Magdalena Morus)
Wykorzystujac PSO oraz biblioteki NumPy i matplotlib wyszukaj minimum 
"globalne" funkcji Ackleya lub Rosenbrocka. Wykorzystujac matplotlib 
przygotuj animacje ruchu czasteczek. 
"""


def rosenbrock(x):
    # works for dim >= 2
    # should always give 0.0 for x = [1., 1., 1., ..., 1.]
    return np.sum((1 - x[:-1]) ** 2) + np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2)


def pso(criterion,
        n_particles=25,
        n_steps=100,
        step_size=1.7,
        vibrations=0.001,
        dim=2):
    # initialization:
    area_min, area_max = -5, 5
    particles_pos = np.random.uniform(area_min, area_max, size=(n_particles, dim))
    # plotting preparation:
    gridsize = 80
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    x = np.linspace(area_min, area_max, gridsize)
    y = np.linspace(area_min, area_max, gridsize)
    X, Y = np.meshgrid(x, y)
    Z = np.empty_like(X)
    for i in range(gridsize):
        for j in range(gridsize):
            Z[j, i] = criterion(np.array((X[j, i], Y[j, i])))
    ax.plot_wireframe(X, Y, Z, color='r', linewidth=0.2)
    # Animation image placeholder
    images = []
    for step in range(n_steps):
        # calculating fitness:
        fitness = [criterion(particles_pos[i]) for i in range(n_particles)]
        best_index = np.argmin(fitness)
        # saving the best individual:
        best = np.copy(particles_pos[best_index])
        print(f'Best fitness after {step} steps: {fitness[best_index]:.4f} '
              f'for the solution: {best}')
        # updating velocities:
        directions = best - particles_pos
        particles_vel = step_size * directions
        # updating positions:
        particles_pos += particles_vel
        # adding vibrations:
        particles_pos += np.random.normal(scale=vibrations,
                                          size=particles_pos.shape)
        particles_pos[best_index, :] = best
        # Add plot for each generation (within the generation for-loop)
        image = ax.scatter3D([
            particles_pos[n][0] for n in range(n_particles)],
            [particles_pos[n][1] for n in range(n_particles)],
            [criterion(particles_pos[n]) for n in
             range(n_particles)], c='b')
        images.append([image])
    # Generate the animation image and save
    animated_image = animation.ArtistAnimation(fig, images, interval=50)
    animated_image.save('./pso_simple.gif', writer='pillow')


if __name__ == '__main__':
    pso(rosenbrock)
