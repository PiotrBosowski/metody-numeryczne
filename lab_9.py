import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import least_squares

x_train = (-1.0077903311937846,
           -0.9082374375674063,
           -0.6878422667350763,
           -0.5806037506675183,
           -0.47237259562526823,
           -0.37928232620964786,
           -0.27181345088635955,
           -0.16289959477712723,
           -0.04985602546516854,
           0.046727327937342356,
           0.1621791985592076,
           0.2711851983707321,
           0.3740720186800417,
           0.47607928546747225,
           0.5778980765002146,
           0.6725757306053213,
           0.7718354397244069,
           0.9011591258913332,
           1.0083557584578497)

y_train = (0.0786572149565985,
           0.1889364731997949,
           0.06242735830288071,
           0.07446362941478313,
           -0.10330041988209748,
           -0.012591227500706292,
           0.07143231102687908,
           0.6070123451619325,
           0.9331277551490533,
           1.1154256934337146,
           2.194135262766615,
           2.2585102038679414,
           1.9106415505272092,
           1.287912421599322,
           0.6062846193313307,
           0.19305152717715712,
           0.21171062689130205,
           0.12536255405589403,
           0.12431023109221773)


def dummy_fn(x: np.ndarray, params: np.ndarray):
    return params[0] \
        * np.exp((-(x - params[1]) ** 2) / (2 * params[2] ** 2))


def criterion(params, criterion=dummy_fn, x=x_train, y=y_train):
    outputs = criterion(x, params)
    return y - outputs


if __name__ == '__main__':
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_0 = np.array([1.0, 2.0, 3.0])
    result = least_squares(criterion, x_0, method='lm')
    print('Success:', result.success)
    print('Best x:', result.x)
    print('Num. of evaluations:', result.nfev)
    ###
    outputs = dummy_fn(x_train, result.x)
    plt.scatter(x_train, y_train, label='Ground-truth')
    plt.scatter(x_train, outputs, label='Best fit')
    plt.legend()
    plt.show()
    dbg_stp = 5
