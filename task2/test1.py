from task1 import mod_seq
try:
    from task import mod_seq_many
except:
    from task import mod_seq_multi as mod_seq_many
import numpy as np

for i in range(10):
    v1 = mod_seq(np.array([1, 1, 1]), np.array([1, 1, 1]), i, 1000000)
    v2 = mod_seq(np.array([2, 2, 2]), np.array([1, 1, 1]), i, 1000000)
    v3 = mod_seq(np.array([1, 1, 1]), np.array([1, 2, 3]), i, 1000000)
    vv1 = mod_seq_many(np.array([1, 1, 1]), np.array([1, 1, 1]), i, 1000000)
    vv2 = mod_seq_many(np.array([[1, 1, 1]]), np.array([1, 1, 1]), i, 1000000)
    vv3 = mod_seq_many(np.array([[1, 1, 1], [2, 2, 2]]), np.array([1, 1, 1]), i, 1000000)
    print(vv1.shape == (), np.all(vv1 == v1),) # нам нужен скаляр

