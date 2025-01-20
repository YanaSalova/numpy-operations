from task import gauss
import numpy as np

def test():
    single = np.array([[2, 6]])

    result = gauss(single)
    if result is None:
        print("fail")
        return
    if type(result.__class__.is_single) != property:
        print("fail")
        return
    if type(result.__class__.fd) != property:
        print("fail")
        return
    if type(result.__class__.solutions) != property:
        print("fail")
        return
    if not result.is_single:
        print("fail")
        return
    if result.fd != 0:
        print("fail")
        return
    try:
        result.is_single = 5
    except:
        pass
    else:
        print("fail")
        return
    try:
        result.fd = 5
    except:
        pass
    else:
        print("fail")
        return
    try:
        result.solutions = 5
    except:
        pass
    else:
        print("fail")
        return
    try:
        del result.is_single
    except:
        pass
    else:
        print("fail")
        return
    try:
        del result.fd
    except:
        pass
    else:
        print("fail")
        return
    try:
        del result.solutions
    except:
        pass
    else:
        print("fail")
        return


    empty = np.array([])
    scalar = np.array(5)
    single = np.array([5])
    single2d = np.array([[5]])
    many = np.arange(100)
    manydims = np.arange(144).reshape(1, 2, 3, 4, 6)
    for bad in scalar, single, single2d, many, manydims:
        try:
            result.solutions(bad)
        except ValueError:
            pass
        else:
            print("fail for shape", bad.shape)

    empty = np.array([])

    values = result.solutions(empty.reshape(0))
    expected = np.array(3)
    if expected.shape != values.shape:
        print("fail scalar")
    if not np.all(expected == values):
        print("fail scalar")

    for i in range(10):
        expected = np.ones((i, 1))
        expected *= 3
        values = result.solutions(empty.reshape(i, 0))
        if expected.shape != values.shape:
            print("fail 1d") 
        if not np.all(expected == values):
            print("fail 1d")

    for i in range(10):
        for j in range(10):
            expected = np.ones((i, j, 1))
            expected *= 3
            values = result.solutions(empty.reshape(i, j, 0))
            if expected.shape != values.shape:
                print("fail 2d")
            if not np.all(expected == values):
                print("fail 2d")

    sh = 2, 3, 123, 15 
    values = result.solutions(empty.reshape(sh + (0, )))
    expected = np.ones(sh + (1,))
    expected *= np.array(3)
    if expected.shape != values.shape:
        print("fail")
    if not np.all(expected == values):
        print("fail")

test()
print("done")

