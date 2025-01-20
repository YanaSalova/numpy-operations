import numpy as np


def mod_seq(seed, coefs, pos, divider):
    n = len(seed)

    if pos < n:
        return seed[-(pos + 1)] % divider

    transition = np.zeros((n, n), dtype=object)
    transition[0] = coefs % divider
    for i in range(1, n):
        transition[i, i - 1] = 1

    def mat_mult(a, b):
        return np.mod(np.dot(a, b), divider)

    def mat_pow(mat, power):
        result = np.identity(n, dtype=object)
        while power > 0:
            if power % 2 == 1:
                result = mat_mult(result, mat)
            mat = mat_mult(mat, mat)
            power = power // 2
        return result

    power = pos - n + 1
    mat_power = mat_pow(transition, power)

    initial = seed[:n] % divider

    pos_element = np.mod(np.dot(mat_power[0], initial), divider)

    return int(pos_element)
