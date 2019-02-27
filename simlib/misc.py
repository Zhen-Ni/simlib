#!/usr/bin/env python3


import numpy as np


def makeplain(name, data):
    """Extract one-dimensional array from nested data structure.

    A block is a component in the control circuit. The input and output of a
    block is updated every sampling time. Each input or output can be a number
    or an array.

    Parameters
    ----------
    name: string
        The name of the data to be extract.

    data: array-like object
        Nested data structure.

    Returns
    ---------
    names: list of strings
        A list of names of extracted data. Each name is the original name of
        data with its item index put in brackets.

    datas: list of extracted data
        Datas is a list of one-dimentional array-like objects.
    """
    if np.iterable(data[0]):
        # data是高维的时间序列
        n = len(data[0])
        names = []
        datas = []
        for i in range(n):
            name_i = "{name}[{i}]".format(name=name, i=i)
            data_i = [j[i] for j in data]
            namesi, datasi = makeplain(name_i, data_i)
            names += namesi
            datas += datasi
    else:
        # 一维时间序列
        names = [name]
        datas = [data]
    return names, datas


def as_uint(ni, positive_number=True, eps=1e-6, msg=""):
    """Convert n to int if available.
    if n is not int or n <(=) 0, raise ValueError."""
    try:
        n = round(ni)
    except TypeError:
        raise ValueError(msg)
    if abs(n - ni) > eps:
        raise ValueError(msg)
    if n < positive_number:
        raise ValueError(msg)
    return n


def tf_dot(coeffs, xs):
    """Calculate result for MIMO system.

    Do convolutiuon between coeffs and xs.

    Example
    -------
    >>> a =np.array( [1,2,3,2,5,0,3,3,4,4,6,1]).reshape([2,2,3])
    >>> a
    array([[[1, 2, 3],
            [2, 5, 0]],

           [[3, 3, 4],
            [4, 6, 1]]])
    >>> b = np.array([1,0,0,2,3,0]).reshape(3,2)
    >>> b
    array([[1, 0],
           [0, 2],
           [3, 0]])
    >>> res = tf_dot(a,b)
    >>> res
    array([16, 25])
    """
    coeffs = np.asarray(coeffs)
    xs = np.asarray(xs)
    # SISO system
    if len(coeffs.shape) == 1:
        if len(xs.shape) != 1:
            raise ValueError('shape of coeffs and xs do not match')
        n = coeffs.shape[0]
        try:
            res = np.dot(coeffs, xs[-1:-1 - n:-1])
        except ValueError:
            if xs.shape[0] < n:
                raise ValueError('length of xs shorter than needed')
            else:
                raise
        return res
    # MIMO system
    elif len(coeffs.shape) == 3:
        if len(xs.shape) != 2 or coeffs.shape[1] != xs.shape[1]:
            raise ValueError('shape of coeffs and xs do not match')
        n_out, n_in, n = coeffs.shape
        try:
            res = (np.dot(coeffs.reshape(n_out, -1),
                          xs[-1:-1 - n:-1].T.reshape(-1)))
        except ValueError:
            if xs.shape[0] < n:
                raise ValueError('length of xs shorter than needed')
            else:
                raise
        return res
    else:
        raise ValueError('dimension of coeffs must be 1 or 3')


def _test_makeplain():
    name = 'array'
    data = np.linspace(0, 80, 81).reshape(9, 3, 3)
    names, datas = makeplain(name, data)
    return names, datas


if __name__ == '__main__':
    _test_makeplain()
