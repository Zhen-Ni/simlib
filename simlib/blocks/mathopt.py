"""mathopt.py

This module contains the basic math operations used in control system.
"""

import numpy as np
import operator

from ..simsys import BaseBlock


__all__ = ['Sum', 'Gain', 'DotProduct']


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
