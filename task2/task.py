import numpy as np


def mod_seq_many(seeds, coefs, pos, divider):
    if not isinstance(seeds, np.ndarray) or not isinstance(coefs, np.ndarray):
        raise TypeError("Параметры seeds и coefs должны быть типа numpy.ndarray.")

    if seeds.shape[-1] != coefs.shape[-1]:
        raise ValueError("Последние размерности seeds и coefs должны совпадать.")

    k = seeds.shape[-1]

    broadcast_shape = np.broadcast_shapes(seeds.shape[:-1], coefs.shape[:-1])

    seeds_b = np.broadcast_to(seeds, broadcast_shape + (k,))
    coefs_b = np.broadcast_to(coefs, broadcast_shape + (k,))

    if pos < k:
        return np.mod(seeds_b[..., -(pos + 1)], divider)

    num_sequences = np.prod(broadcast_shape) if broadcast_shape else 1

    if num_sequences > 0:
        seeds_flat = seeds_b.reshape((num_sequences, k))
        coefs_flat = coefs_b.reshape((num_sequences, k))

        transition = np.zeros((num_sequences, k, k), dtype=object)
        transition[:, 0, :] = np.mod(coefs_flat, divider)
        for i in range(1, k):
            transition[:, i, i - 1] = 1

        mat_powers = np.array([np.identity(k, dtype=object)] * num_sequences)
        power = pos - k + 1

        while power > 0:
            if power % 2 == 1:
                mat_powers = mat_mult(transition, mat_powers, divider)
            transition = mat_mult(transition, transition, divider)
            power = power // 2

        initial = seeds_flat[..., ::-1] % divider
        pos_elements = np.mod(np.sum(mat_powers[:, 0, :] * initial, axis=1), divider)

        if broadcast_shape:
            return pos_elements.reshape(broadcast_shape)
        else:
            return pos_elements[0]
    else:
        return np.array([])


def mat_mult(a, b, divider):
    return np.mod(np.matmul(a, b), divider)
