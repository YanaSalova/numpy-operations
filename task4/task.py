import numpy as np


def mod_seq_many(pairs: np.ndarray):
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("Входной массив должен иметь форму (n, 2).")
    if pairs.dtype != np.uint64:
        raise ValueError("Тип элементов массива должен быть np.uint64.")
    a = pairs[:, 0].copy()
    b = pairs[:, 1].copy()
    while np.any(b != 0):
        mask = b != 0
        a[mask], b[mask] = b[mask], a[mask] % b[mask]
    return a
