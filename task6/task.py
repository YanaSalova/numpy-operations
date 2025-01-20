import numpy as np
import operator


class ExtNdArray:
    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a NumPy ndarray")
        self._data = data

    @staticmethod
    def extended_broadcast_operation(op, data1, data2):
        shape1 = np.array(data1.shape)
        shape2 = np.array(data2.shape)
        ndim = max(len(shape1), len(shape2))
        shape1 = np.pad(shape1, (ndim - len(shape1), 0), constant_values=1)
        shape2 = np.pad(shape2, (ndim - len(shape2), 0), constant_values=1)
        result_shape = np.maximum(shape1, shape2)
        if np.any((result_shape % shape1 != 0) & (shape1 != 1)) or np.any(
            (result_shape % shape2 != 0) & (shape2 != 1)
        ):
            raise ValueError("Shapes are not compatible for extended broadcasting")
        idx1 = [np.arange(s) % d for s, d in zip(result_shape, shape1)]
        idx2 = [np.arange(s) % d for s, d in zip(result_shape, shape2)]
        grid1 = np.ix_(*idx1)
        grid2 = np.ix_(*idx2)
        return op(data1[grid1], data2[grid2])

    @staticmethod
    def _binary_operation(op, a, b):
        if isinstance(a, ExtNdArray):
            a = a._data
        if isinstance(b, ExtNdArray):
            b = b._data
        result = ExtNdArray.extended_broadcast_operation(op, a, b)
        return ExtNdArray(result)

    @classmethod
    def _generate_binary_methods(cls):
        ops = {
            "__add__": operator.add,
            "__radd__": operator.add,
            "__sub__": operator.sub,
            "__rsub__": lambda a, b: operator.sub(b, a),
            "__mul__": operator.mul,
            "__rmul__": operator.mul,
            "__truediv__": operator.truediv,
            "__rtruediv__": lambda a, b: operator.truediv(b, a),
        }
        for name, op in ops.items():
            setattr(
                cls,
                name,
                lambda self, other, op=op: cls._binary_operation(op, self, other),
            )

    def reshape(self, *args, **kwargs):
        reshaped = self._data.reshape(*args, **kwargs)
        return ExtNdArray(reshaped)

    @property
    def shape(self):
        return self._data.shape

    @shape.setter
    def shape(self, new_shape):
        self._data.shape = new_shape

    @property
    def ndim(self):
        return self._data.ndim

    def __getattr__(self, name):
        return getattr(self._data, name)

    def __array__(self):
        return self._data


ExtNdArray._generate_binary_methods()
