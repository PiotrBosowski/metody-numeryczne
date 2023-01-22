import numpy as np
import matplotlib.pyplot as plt
plt.ion()


G = 6.6743e-11

# MASSES = np.array([1.98847e30, 5.97219e24, 6.39e23, 1.8982 * 10 ** 27, 3.3011 * 10 ** 23])  # kg
# positions_0 = np.array([[0., 0, 0], [149598023 * 10 ** 3, 0., 0], [227939366 * 10 ** 3, 0., 0], [740800000 * 10 ** 3, 0., 0], [57909050 * 10 ** 3, 0., 0]])
# velocities_0 = np.array([[0., 0, 0], [0., 29.78 * 10 ** 3, 0], [0., 24.07 * 10 ** 3, 0], [0., 13.07 * 10 ** 3, 0], [0., 47.36 * 10 ** 3 , 0]])  # km/s

# reduced version for debug purposes:
MASSES = np.array([1.98847e30, 5.97219e24, 1.8982 * 10 ** 27])  # kg
positions_0 = np.array([[0., 0, 0], [149598023 * 10 ** 3, 0., 0], [227939366 * 10 ** 3, 0., 0]])
velocities_0 = np.array([[0., 0, 0], [0., 29.78 * 10 ** 3, 0], [0., 24.07 * 10 ** 3, 0]])  # m/s
del_t = 60. * 60 * 24 * 5

t = 0
positions = positions_0
velocities = velocities_0
I = []
J = []
n = MASSES.size
tf = 1. * 360 * 60. * 60 * 24 * 5

for i in range(n):
    for j in range(n):
        if i != j:
            I.append(i)
            J.append(j)

I = np.array(I)
J = np.array(J)
Nint = I.size

def F(R):
    dr = np.sum((R[I] - R[J]) ** 2, axis=1) ** 0.5
    dr = dr.reshape(Nint, 1)
    er = (R[I] - R[J]) * dr ** -1
    F = -(G * MASSES[I] * MASSES[J]).reshape(Nint, 1) * dr ** -2 * er
    F = F.reshape(n, n - 1, 3)
    F = np.sum(F, axis=1)
    return F

fig = plt.figure()
ax = fig.add_subplot(projection="3d")


def get_next_state(positions, velocities, delta_t):
    accelerations = F(positions) * MASSES.reshape(n, 1) ** -1
    velocities = velocities + accelerations * delta_t
    positions = positions + velocities * delta_t + 0.5 * accelerations * delta_t ** 2
    return positions, velocities


while tf > t:
    t += del_t
    positions, velocities = get_next_state(positions, velocities, del_t)
    after_positions, after_velocities = get_next_state(positions, velocities, del_t)
    positions = (positions + after_positions) / 2.
    velocities = (velocities + after_velocities) / 2.
    # for debug purposes:
    # ax.cla()
    ax.set(xlim3d=(-1.2 * np.max(positions_0), 1.2 * np.max(positions_0)), xlabel='X')
    ax.set(ylim3d=(-1.2 * np.max(positions_0), 1.2 * np.max(positions_0)), ylabel='Y')
    ax.set(zlim3d=(-1.2 * np.max(positions_0), 1.2 * np.max(positions_0)), zlabel='Z')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2])
    plt.pause(0.01)

plt.show()
