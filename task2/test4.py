from task1 import mod_seq
try:
    from task import mod_seq_many
except:
    from task import mod_seq_multi as mod_seq_many
import numpy as np

for i in range(10):
    v1 = mod_seq(np.array([1, 2, 3]), np.array([-1, -2, -3]), i, 10)
    v2 = mod_seq(np.array([1, 2, 3]), np.array([2, 4, 6]), i, 10)
    vv1 = mod_seq_many(np.array([1, 2, 3]), np.array([[-1, -2, -3], [2, 4, 6]]), i, 10)
    print(vv1.shape == (2,), np.all(vv1 == np.array([v1, v2])),)

