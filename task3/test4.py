from task import gauss
import numpy as np

def test():
    single = np.array([[1, 2], [2, 5]])

    result = gauss(single)
    if result is not None and hasattr(result, 'solutions'):
        print("fail")
        return

test()
print("done")
