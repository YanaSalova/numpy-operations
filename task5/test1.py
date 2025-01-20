from task import extended_bcast_add
import numpy as np

nodata = np.array([])
scalar = np.array(12)
single = np.array([345])
single2d = np.array([[6789]])
dim_1_2 = np.array([[34], [5]])
crazy = np.array([55]).reshape((1,) * 32)
more_crazy = np.array(range(2 * 3 * 4 * 5 * 6)).reshape(tuple(range(1, 7))[::-1])

zoo = nodata, scalar, single, single2d, dim_1_2, crazy, more_crazy

for e1 in zoo:
    for e2 in zoo:
        v1 = e1 + e2
        v2 = extended_bcast_add(e1, e2)
        if not np.all(v1 == v2):
            print("fail") 
print("done")

