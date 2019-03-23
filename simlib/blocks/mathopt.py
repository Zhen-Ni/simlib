"""mathopt.py

This module contains the basic math operations used in control system.
"""

import numpy as np
import operator

from ..simsys import BaseBlock


__all__ = ['Sum', 'Gain', 'DotProduct', 'UserDefinedFunction']


class Sum(BaseBlock):

    """The Sum block.

    Add or abstract inputs.

    Parameters
    ----------
    operators: string
        String contains '+' or '-'. (default = '++')

    name: string, optional
        The name of this block.
    """

    def __init__(self, operators='++', name='Sum'):
        self._opts = []
        for i in operators:
            if i == '+':
                self._opts.append(operator.add)
            elif i == '-':
                self._opts.append(operator.sub)
            else:
                raise ValueError('operator must be iterable with "+" or "-"')
        super().__init__(nin=len(self._opts), nout=1, name=name)

    def BLOCKSTEP(self, *inSig):
        res = 0.0
        for i in range(len(inSig)):
            res = self._opts[i](res, inSig[i])
        return res,


class Gain(BaseBlock):

    """The Gain block.

    Parameters
    ----------
    k: float
        Cofficient for gain. (default = 1.0)

    name: string, optional
        The name of this block.
    """

    def __init__(self, k=1.0, name='Gain'):
        super().__init__(nin=1, nout=1, name=name)
        self._k = k

    def BLOCKSTEP(self, *xs):
        res = xs[0] * self._k
        return res,


class DotProduct(BaseBlock):

    """The DotProduct block.

    Calculate product between inputs. The product operation is defined by
    the '*' operator of the inputs.

    Parameters
    ----------
    nin: int, optional
        Number of input signals. (default = 2)

    name: string, optional
        The name of this block.
    """

    def __init__(self, nin=2, name='Dot Product'):
        super().__init__(nin=nin, nout=1, name=name)

    def BLOCKSTEP(self, *xs):
        res = 1.0
        for i in range(len(xs)):
            res *= xs[i]
        return res,


class UserDefinedFunction(BaseBlock):
    """The UserDefinedFunction block.

    This block runs the user defined function in each iteration.

    Parameters
    ----------
    func: callable
        The user defined function processing the input signals. The number of 
        inputs to the function is defined by nin. The output of the function 
        can be iterable and the length should be given by nout, and each
        element in the iterable object will be sent to the outports. However,
        if nout is set to None, the output of the function is not necessarily
        iterable and it will be sent to outports[0].

    t_as_arg: bool, optional
        Whether the user-defined function `func` accepts `t` as a keyword
        argument. If it is True, the simulation time will be an additional
        keyword argument passed to the function within each iteration. (default
        to False)

    nin: int, optional
        Number of input signals. This should also be the number of input 
        parameters of func. (default to 1)

    nout: int or None, optional
        Number of output signals. If nout is None, output of the user-defined
        function will be sent to outports[0].
        (default = None)

    dt: float, optional
        Sampling time of this block. The input and output be evaluate once
        for each sampling time.

    name: string, optional
        The name of this block.
    """

    def __init__(self, func, t_as_arg=False, nin=1, nout=None, dt=None,
                 name='UserDefinedFunction'):
        if nout is None:
            super().__init__(nin=nin, nout=1, dt=dt, name=name)
        else:
            super().__init__(nin=nin, nout=nout, dt=dt, name=name)

        if t_as_arg:
            self._func = lambda _t, *_xs: func(*_xs, t=_t)
        else:
            self._func = lambda _t, *_xs: func(*_xs)
        self._direct_output = False if nout is None else True

    def BLOCKSTEP(self, *xs):
        if self._direct_output:
            return self._func(self.t, *xs)
        else:
            return self._func(self.t, *xs),
