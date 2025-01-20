from task import extended_bcast_add
import numpy as np

for t in np.uint64, np.uint32, np.uint8, np.float32, np.float64, np.int8:
    data1 = np.array(list(range(2 * 3 * 4 * 5)), dtype=t)

    for i in 1, 2, 3, 4, 5, 120:
        data2 = (np.arange(i, dtype=t) + 1) * 3 
        data3 = np.concatenate([data2] * (120 // i))
        exp = data1 + data3
        result1 = extended_bcast_add(data1, data2)
        result2 = extended_bcast_add(data2, data1)
        if result1.dtype != exp.dtype:
            print("fail")
        if result2.dtype != exp.dtype:
            print("fail")
        if not np.all(result1 == exp):
            print("fail")
        if not np.all(result2 == exp):
            print("fail")


print("done")

