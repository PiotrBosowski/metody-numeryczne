import time

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize


def nacl_potential(r):
    r_0 = 0.33
    V_0 = 1.09E3
    value = V_0 * np.exp(-r / r_0) + \
            np.divide(-(np.e ** 2), (4 * np.pi * r),
                      out=np.zeros_like(r),
                      where=r != 0)
    return np.triu(value, k=1)


def lennard_jones_potential(r):
    r_triu = np.triu(r, k=1)
    sigma = 1.0
    eps = 5.0
    sigma_r_ratio = np.divide(sigma, r_triu,
                              out=np.zeros_like(r_triu),
                              where=r_triu != 0)
    return 4. * eps * (sigma_r_ratio ** 12 - sigma_r_ratio ** 6)


def fill_cube(num_of_atoms, cube_size=1.):
    """
    Generates a matrix of coordinates of num_of_atoms randomly placed
    (uniform distribution) atoms. Each row describes 3 coordinates of a
    particular atom: x, y, and z. All atoms are uniformly distributed
    within a 0-cube_size 3-D cube.
    :param num_of_atoms: number of atoms to be placed in a cube
    :param cube_size: size of a cube to contain all the atoms
    :return: Coordinate matrix: [(x1, y1, z1), ...]
    """
    return np.random.uniform(low=0, high=cube_size,
                             size=(num_of_atoms, 3))


def get_total_potential(coords_matrix, potential_fn):
    """
    Calculate a sum of all potentials between particles in the system,
    described by the coords_matrix, according to the potential_function,
    which defines a potential between two individual particles.
    :param potential_fn:
    :param coords_matrix:
    :return: A total potential value in the system.
    """
    num_of_atoms = coords_matrix.shape[0]
    dist_tensor = np.empty((num_of_atoms, num_of_atoms, 3))  # 3 for 3 coords
    for i, j in np.ndindex((num_of_atoms, num_of_atoms)):
        dist_tensor[i, j, :] = coords_matrix[i] - coords_matrix[j]
    squared_dist_tensor = dist_tensor ** 2
    squared_dist_matrix = np.sum(squared_dist_tensor, axis=2)
    dist_matrix = np.sqrt(squared_dist_matrix)
    potentials_u = potential_fn(dist_matrix)
    return np.sum(potentials_u)


def plot_coordinates(coords_matrix):
    c = coords_matrix
    plt.style.use('seaborn-poster')
    ax = plt.axes(projection='3d')
    ax.grid()
    ax.scatter3D(c[:, 0], c[:, 1], c[:, 2])
    plt.show()


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
                              nacl_potential, method=method)
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
