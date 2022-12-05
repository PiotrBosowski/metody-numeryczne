def f(x):
    return x**3 - 1500000 * x**2 + 750000000002*x - 125000000000999990


def regula_falsi(criterion, a, b, eps_1=0.1, eps_2=0.1, iter_max=10000):
    if criterion(a) * criterion(b) > 0:
        raise RuntimeError("A criterion function does not satisfy the "
                           "requirements.")
    old_c = c = b
    for iteration in range(iter_max):
        crit_a = criterion(a)
        crit_b = criterion(b)
        c = (a * crit_b - b * crit_a) /\
            (crit_b - crit_a)
        if abs(old_c - c) < eps_2:
            break
        crit_c = criterion(c)
        if abs(crit_c) < eps_1:
            break
        if criterion(a) * crit_c > 0:
            a = c
        else:
            b = c
        old_c = c
    return c


print(regula_falsi(f, 0, 1E6, iter_max=124124))
