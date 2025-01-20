from task import extended_bcast_add
import numpy as np

data1 = np.array(list(range(2 * 3 * 4 * 5)), dtype=np.uint64)

for rb1 in range(10):
    shb1 = (1,) * rb1
    for rb2 in range(10):
        shb2 = (1,) * rb2
        for ra in range(10):
            sha = (1,) * ra
            data1 = data1.reshape(shb1 + (120,) + sha)
            for i in 1, 2, 3, 4, 5, 120:
                data2 = (np.arange(i, dtype=np.uint64) + 1) * 3 
                data3 = np.concatenate([data2] * (120 // i))
                data2 = data2.reshape(shb2 + data2.shape + sha)
                data3 = data3.reshape(shb2 + data3.shape + sha)
                exp = data1 + data3
                result1 = extended_bcast_add(data1, data2)
                result2 = extended_bcast_add(data2, data1)
                if not np.all(result1 == exp):
                    print("fail")
                if not np.all(result2 == exp):
                    print("fail")

print("done")

