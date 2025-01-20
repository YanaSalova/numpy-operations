import numpy as np


def extended_bcast_add(data1, data2):
    try:
        return data1 + data2
    except ValueError:
        shape1 = np.array(data1.shape, dtype=int)
        shape2 = np.array(data2.shape, dtype=int)
        ndim = max(len(shape1), len(shape2))
        shape1 = np.pad(shape1, (ndim - len(shape1), 0), constant_values=1)
        shape2 = np.pad(shape2, (ndim - len(shape2), 0), constant_values=1)
        result_shape = np.where(
            (shape1 == 0) | (shape2 == 0), 0, np.maximum(shape1, shape2)
        ).astype(int)
        with np.errstate(divide='ignore', invalid='ignore'):
            mod1 = np.where(
                (result_shape != 0) & (shape1 > 1),
                (result_shape % shape1) == 0,
                True
            )
            mod2 = np.where(
                (result_shape != 0) & (shape2 > 1),
                (result_shape % shape2) == 0,
                True
            )
        valid1 = (shape1 == 1) | (shape1 == result_shape) | mod1
        valid2 = (shape2 == 1) | (shape2 == result_shape) | mod2
        valid1 &= ~((shape1 == 0) & (result_shape != 0))
        valid2 &= ~((shape2 == 0) & (result_shape != 0))
        if not np.all(valid1) or not np.all(valid2):
            raise ValueError("Shapes are not compatible for broadcasting")
        try:
            data1 = data1.reshape(tuple(shape1))
        except Exception as e:
            raise ValueError(f"Reshaping data1 failed: {e}")

        try:
            data2 = data2.reshape(tuple(shape2))
        except Exception as e:
            raise ValueError(f"Reshaping data2 failed: {e}")

        idx1 = []
        idx2 = []
        for s, d in zip(result_shape, shape1):
            if d == 0:
                idx1.append(np.array([], dtype=int))
            elif d == 1:
                idx1.append(np.zeros(s, dtype=int))
            else:
                idx1.append(np.arange(s) % d)

        for s, d in zip(result_shape, shape2):
            if d == 0:
                idx2.append(np.array([], dtype=int))
            elif d == 1:
                idx2.append(np.zeros(s, dtype=int))
            else:
                idx2.append(np.arange(s) % d)
        if np.any(result_shape == 0):
            return np.empty(result_shape, dtype=np.result_type(data1, data2))
        if ndim > 0:
            try:
                grid1 = np.ix_(*idx1)
                grid2 = np.ix_(*idx2)
            except ValueError:
                return np.empty(result_shape, dtype=np.result_type(data1, data2))
        else:
            grid1 = ()
            grid2 = ()
        if ndim == 0:
            result = data1 + data2
        else:
            try:
                result = data1[grid1] + data2[grid2]
            except Exception as e:
                raise ValueError(f"Element-wise addition failed: {e}")
        return result
