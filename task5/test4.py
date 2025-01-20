from task import extended_bcast_add
import numpy as np

data1 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.uint64)
data2 = np.array([[12, 23, 34, 45, 56, 67], [123, 234, 345, 456, 567, 678]], dtype=np.uint64)

expected = [
 [13,  25,  35,  47,  57,  69],
 [126, 238, 348, 460, 570, 682],
 [ 17,  29,  39,  51,  61,  73],
 [130, 242, 352, 464, 574, 686],
]
expected = np.array(expected, dtype=np.uint64)
result = extended_bcast_add(data1, data2)
if not np.all(expected == result):
    print("fail")

print("done")

