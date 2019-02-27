#!/usr/bin/env python3


import numpy as np

from ..simsys import BaseBlock
from ..simexceptions import SimulationError

__all__ = ['Unit', 'Mux', 'Demux', 'Bundle']


class Unit(BaseBlock):

    """Do nothing but output its input signal.

    Parameters
    ----------
     dt: float, optional
        Sampling time.

    name: string, optional
        Name of this block. (default = 'Delay')

    See Also
    ----------
    BaseBlock
    """

    def __init__(self, dt=None, name='Unit'):
        super().__init__(nin=1, nout=1, dt=dt, name=name)

    def BLOCKSTEP(self, *xs):
        return xs


class Mux(BaseBlock):

    """Multiplex scalar or vector signals.

    Note that signal dimension larger than one will be converted to
    one-dimensional.

    Parameters
    ----------
    nin: int
        Number of input signals. (default = 2)

    name: string, optional
        Name of this block. (default = 'Mux')

    See Also
    ----------
    BaseBlock
    """

    def __init__(self, nin=2, name='Mux'):
        if nin >= 0.5:
            nin = round(nin)
        else:
            raise ValueError('nin must be greater than 0')
        super().__init__(nin=nin, nout=1, name=name)

    def BLOCKSTEP(self, *xs):
        signals = []
        for s in xs:
            if not np.iterable(s):
                signals.append([s])
            else:
                signals.append(np.asarray(s, dtype=object).reshape(-1))
        res = np.concatenate(signals)
        return res,


class Demux(BaseBlock):

    """Split signal into scalars or arrays with smaller dimensions.

    Note that Demux is not the inverse of Mux if dimension of signal is larger
    than 1.

    Parameters
    ----------
    nout: int
        Number of output signals. (default = 2)

    default: optional
        Defalt value for output signal. This value will be output if size of
        input signal is smaller than nout. If default = None and the sizes do
        not match, SimulationError will be raised. (default = None)

    name: string, optional
        Name of this block. (default = 'Demux')

    See Also
    ----------
    BaseBlock
    """

    def __init__(self, nout=2, default=None, name='Demux'):
        if nout >= 0.5:
            nout = round(nout)
        else:
            raise ValueError('nout must be greater than 0')
        self._default = default
        super().__init__(nin=1, nout=nout, name=name)

    def BLOCKSTEP(self, *xs):
        nout = len(self._outports)
        if np.iterable(xs[0]):
            signals = [i for i in xs[0]]
        else:
            signals = [xs[0]]

        if len(signals) == nout:
            return signals
        elif len(signals) < nout and self._default is not None:
            res = [self._default] * nout
            res[:len(signals)] = signals
            return res
        else:
            raise SimulationError("dimension input signal (={a}) to {s} does "
                                  "not match the block's number of outputs "
                                  "(={b})"
                                  .format(s=self, a=len(signals), b=nout))


class Bundle(BaseBlock):

    """Combine signals.

    Bundle signals from all inports into an iterable object.

    Parameters
    ----------
    nin: int
        Number of input signals. (default = 2)

    name: string, optional
        Name of this block. (default = 'Bundle')

    See Also
    ----------
    BaseBlock
    """

    def __init__(self, nin=2, name='Bundle'):
        if nin >= 0.5:
            nin = round(nin)
        else:
            raise ValueError('nin must be greater than 0')
        super().__init__(nin=nin, nout=1, name=name)

    def BLOCKSTEP(self, *xs):
        signals = []
        for s in xs:
            signals.append(s)
        return np.array(signals),


class Unbundle(BaseBlock):

    """Seperate signals.

    Unbundle signals from an iterable object.

    Parameters
    ----------
    nout: int
        Number of output signals. (default = 2)

    default: optional
        Defalt value for output signal. This value will be output if size of
        input signal is smaller than nout. If default = None and the sizes do
        not match, SimulationError will be raised. (default = None)

    name: string, optional
        Name of this block. (default = 'Unbundle')

    Ports
    -----
    In[0]:
        The input signal, should be iterable.

    Out[i]:
        The output signals
    """

    def __init__(self, nout=2, default=None, name='Demux'):
        if nout >= 0.5:
            nout = round(nout)
        else:
            raise ValueError('nout must be greater than 0')
        self._default = default
        super().__init__(nin=1, nout=nout, name=name)

    def BLOCKSTEP(self, *xs):
        nout = len(self._outports)
        if np.iterable(xs[0]):
            signals = [i for i in xs[0]]
        else:
            raise SimulationError('input signal to {s} is not iterable'
                                  .format(s=self))

        if len(signals) == nout:
            return signals
        elif len(signals) < nout and self._default is not None:
            res = [self._default] * nout
            res[:len(signals)] = signals
            return res
        else:
            raise SimulationError("dimension input signal (={a}) to {s} does "
                                  "not match the block's number of outputs "
                                  "(={b})"
                                  .format(s=self, a=len(signals), b=nout))
