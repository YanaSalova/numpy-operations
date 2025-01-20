from task import gauss
import numpy as np

def test():
    single = np.array([[2, 1, 6]])
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
    if result.is_single:
        print("fail")
        return
    if result.fd != 1:
        print("fail fd")
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

    for v in range(10):
        s = result.solutions(np.array([v]))
        expected = np.array([(6 - v) / 2, v])
        if expected.shape != s.shape:
            print("fail empty")
        if not np.all(expected == s):
            print("fail empty")

    s = result.solutions(np.array([]).reshape(0, 1))
    expected = np.array([]).reshape(0, 2)
    if expected.shape != s.shape:
        print("fail empty")
    if not np.all(expected == s):
        print("fail empty")

    for v in range(10):
        s = result.solutions(np.array([[v]]))
        expected = np.array([[(6 - v) / 2, v]])
        if expected.shape != s.shape:
           print("fail shape")
        if not np.all(expected == s):
           print("fail result") 

    inp = np.arange(10).reshape(10, 1)
    s = result.solutions(inp)
    exp1 = (6 - inp[:, 0]) / 2 
    exp2 = inp[:, 0]
    expected = np.concatenate([exp1, exp2]).reshape(2, 10).T
    if expected.shape != s.shape:
        print("fail shape")
    if not np.all(expected == s):
        print("fail result")


    inp = np.arange(72).reshape(3, 1, 2, 4, 3, 1)
    s = result.solutions(inp)
    exp1 = (6 - inp.reshape(72)) / 2 
    exp2 = inp.reshape(72)
    expected = np.concatenate([exp1, exp2]).reshape(2, 72).T.reshape(3, 1, 2, 4, 3, 2)
    if expected.shape != s.shape:
        print("fail shape")
    if not np.all(expected == s):
        print("fail result")

    empty = np.array([])
    scalar = np.array(5)
    pair = np.array([5, 2])
    many = np.arange(100)
    manydims = np.arange(144).reshape(1, 2, 3, 4, 6)
    for bad in empty, scalar, pair, many, manydims:
        try:
            result.solutions(bad)
        except ValueError:
            pass
        else:
            print("fail for shape", bad.shape)

test()
print("done")

