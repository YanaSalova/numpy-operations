from task import gauss
import numpy as np

def test():
    single = np.array([[0, 0, 1, 1], [0, 0, 1, 2], [1, 1 , 1, 5]])

    result = gauss(single)
    if result is not None and hasattr(result, 'solutions'):
        print("fail")
        return

test()
print("done")
