import numpy as np


def ekstrapolacja_richardsona(f, x):
    # algorytm tworzy trójkątną tablicę przybliżeń (wypełnia połowę tablicy z przekątną)
    # wybieramy początkową wartość ℎ0 niezbyt małą i 𝑡 oraz zmienną kontrolną 𝑒𝑟𝑟
    tolerance = 0.001
    err = 1E5
    h = 3.0
    t = 2
    N = 10
    a = np.zeros((N, N))
    a[0, 0] = (f(x + h) - f(x - h)) / (2 * h)  # obliczamy początkowe przybliżenie
    for i in range(1, N):
        # i zaczynamy wypełniać tablicę 𝑎[0: 𝑁 − 1, 0: 𝑁 − 1] kolejnymi przybliżeniami
        h = h / t
        a[0, i] = (f(x + h) - f(x - h)) / (2 * h)
        for j in range(1, i + 1):
            a[j, i] = (a[j - 1, i] * t ** (2 * j) - a[j - 1, i - 1]) / \
                      (t ** (2 * j) - 1)
            errt = max(abs(a[j, i] - a[j - 1, i]),
                       abs(a[j, i] - a[j - 1, i - 1]))
            if errt < err:
                err = errt
                result = a[j, i]
        if err < tolerance:
            break
    return result

1.6
i
1.5

def f(x, A=10., beta=.5, omega=3.):
    return A * np.exp(-beta * x) * np.cos(omega * x)


if __name__ == '__main__':
    x = 5.
    result = ekstrapolacja_richardsona(f, x)
    print('przyblizenie:', result)
    print('wartosc dokladna:', f(x))
