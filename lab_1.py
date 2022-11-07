import matplotlib.pyplot as plt
import numpy as np


def float_comparison(initial_number=1E20):
    for float_type in [np.float64, np.float32, np.float16, float]:
        x = float_type(initial_number)
        dx = float_type(initial_number)
        iteration = 0
        while x != x + dx:
            iteration += 1
            dx = dx / float_type(2)
        print(f'It took {iteration} steps to beak numerical precision for '
              f'{float_type}')


def tower_of_coins():
    d = 0.5  # cm
    i, l_c, l_p = 1, 1., 0.
    while l_c > l_p:  # sprawdzamy czy pętla działa poprawnie
        l_p = l_c
        l_c += d / i
        i += 1E8
    print('iter: ', i, ', wysunięcie: ', l_c, ', dodane wysunięcie: ', d / i)
    print('d/i == 0: ',
          d/i == 0)
    print('0.9999999999999999 == 0.9999999999999999 + d/i: ',
          0.9999999999999999 == 0.9999999999999999 + d/i)
    print('1 == 1 + d/i:', 1 == 1 + d/i)


def c14():
    lbd = 1. / 8266.6426  # yr**-1 stała zaniku C-14
    delta_t = 1000.  # yr krok iteracji
    N0 = 1E30  # początkowa liczba atomów C-14
    N = N0
    i, N_p = 0, N * 2
    while N < N_p:  # sprawdzamy czy N maleje wraz z czasem
        N_p = N
        # N -= N * lbd * delta_t
        # zapis rownowazny:
        N *= 1 - (lbd * delta_t)
        # ciag jest geometryczny zbiezny o kroku +/- 7/8
        i += 1
        print('iteracja: ', i, ', N(t): ', N)


def anal_vs_iter():
    lbd = 1. / 8266.6426  # yr**-1 stała zaniku C-14
    delta_t = 2000.  # yr krok iteracji
    N_0 = 1.  # początkowa liczba radionuklidów
    time = np.arange(0., 50000., delta_t)
    N_iter = np.zeros(time.size)
    N_a = N_0 * np.exp(-lbd * time)  # rozwiązanie analityczne
    N = N_0  # ustawiamy początkową liczbę radionuklidów dla metody iteracyjnej
    for i, t in enumerate(time):
        N_iter[i] = N
        N -= N * lbd * delta_t
    _, ax = plt.subplots()
    ax.plot(time, N_iter, '.', label='Rozwiązanie iteracyjne')
    ax.plot(time, N_a, '.', label='Rozwiązanie analityczne')
    ax.plot(time, N_a - N_iter, '.', label='N_a-N_iter')
    ax.set(xlabel='time (yr)', ylabel='N')
    plt.legend()
    plt.show()


def system_2_4(iterations=2500, t=0.001):
    lambda_AB, lambda_AC, lambda_BD, lambda_CD = 1, 2, 3, 4
    N_A0, N_B0, N_C0, N_D0 = 1, 0, 0, 0
    A, B, C, D = [N_A0], [N_B0], [N_C0], [N_D0]
    for i in range(iterations - 1):
        dA = t * (-lambda_AB * A[-1] - lambda_AC * A[-1])
        dB = t * (lambda_AB * A[-1] - lambda_BD * B[-1])
        dC = t * (lambda_AC * A[-1] - lambda_CD * C[-1])
        dD = t * (lambda_BD * B[-1] + lambda_CD * C[-1])
        A.append(A[-1] + dA)
        B.append(B[-1] + dB)
        C.append(C[-1] + dC)
        D.append(D[-1] + dD)
    x = [t * i for i in range(iterations)]
    plt.scatter(x, A, marker='.', label='A')
    plt.scatter(x, B, marker='.', label='B')
    plt.scatter(x, C, marker='.', label='C')
    plt.scatter(x, D, marker='.', label='D')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # float_comparison(2)
    # tower_of_coins()
    # c14()
    # anal_vs_iter()
    system_2_4()
