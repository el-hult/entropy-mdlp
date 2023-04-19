import entropymdlp as mdlp
import numpy as np


def test1():
    """Single Split"""
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    cp = mdlp.cut_points(x, y)
    np.testing.assert_array_equal(cp, [3.5])
    np.testing.assert_array_equal(np.digitize(x, cp), [0, 0, 0, 1, 1, 1, 1, 1, 1, 1])


def test2():
    """Recurse Once Left"""
    x = np.array(
        [1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
    )
    y = np.array(
        [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )

    cp = mdlp.cut_points(x, y)
    np.testing.assert_array_equal(cp, [2.5, 8.5])
    np.testing.assert_array_equal(
        np.digitize(x, cp),
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    )


def test4():
    """Recurse Once Right"""
    y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    x = np.arange(len(y))

    cp = mdlp.cut_points(x, y)
    np.testing.assert_array_equal(cp, [6.5])
    np.testing.assert_array_equal(
        np.digitize(x, cp), [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    )


def test3():
    """Class change on last index"""
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([0, 0, 0, 0, 1])

    cp = mdlp.cut_points(x, y)
    np.testing.assert_array_equal(cp, [4.5])
    np.testing.assert_array_equal(np.digitize(x, cp), [0, 0, 0, 0, 1])


if __name__ == "__main__":
    test1()
    test2()
    test3()
    test4()
