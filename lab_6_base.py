import numpy as np
import matplotlib.pyplot as plt


# Słońce | Ziemia | Mars | Jowisz | Merkury |
G = 6.6743e-11
masses = np.array([1.98847e30, 5.97219e24, 6.39e23, 1.8982 * 10 ** 27, 3.3011 * 10 ** 23])  # kg
positions_0 = np.array([[0., 0, 0], [149598023 * 10 ** 3, 0., 0], [227939366 * 10 ** 3, 0., 0], [740800000 * 10 ** 3, 0., 0], [57909050 * 10 ** 3, 0., 0]])
velocities_0 = np.array([[0., 0, 0], [0., 29.78 * 10 ** 3, 0], [0., 24.07 * 10 ** 3, 0], [0., 13.07 * 10 ** 3, 0], [0., 47.36 * 10 ** 3 , 0]])  # km/s
del_t = 60. * 60 * 24 * 7

t = 0
positions = positions_0
velocities = velocities_0
I = []
J = []
n = masses.size
tf = del_t * 52 * 4

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
    F = -(G * masses[I] * masses[J]).reshape(Nint, 1) * dr ** -2 * er
    F = F.reshape(n, n - 1, 3)
    F = np.sum(F, axis=1)
    return F

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

while tf > t:
    t += del_t
    A = F(positions) * masses.reshape(n, 1) ** -1
    print(A)
    # velocities = velocities + (A[1, :] + A[:, -1]) / 2 * del_t
    velocities = velocities + A * del_t
    positions = positions + velocities * del_t + 0.5 * A * del_t ** 2
    ax.cla()
    ax.set(xlim3d=(-1.2 * np.max(positions_0), 1.2 * np.max(positions_0)), xlabel='X')
    ax.set(ylim3d=(-1.2 * np.max(positions_0), 1.2 * np.max(positions_0)), ylabel='Y')
    ax.set(zlim3d=(-1.2 * np.max(positions_0), 1.2 * np.max(positions_0)), zlabel='Z')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2])
    plt.pause(0.01)

plt.show()