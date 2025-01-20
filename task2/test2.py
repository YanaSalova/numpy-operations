from task1 import mod_seq
try:
    from task import mod_seq_many
except:
    from task import mod_seq_multi as mod_seq_many
import numpy as np

for i in range(10):
    v1 = mod_seq(np.array([1, 1, 1]), np.array([1, 1, 1]), i, 1000000)
    vv1 = mod_seq_many(np.array([[1, 1, 1]]), np.array([1, 1, 1]), i, 1000000)
    print(vv1.shape == (1,), np.all(vv1 == np.array([v1])),)

