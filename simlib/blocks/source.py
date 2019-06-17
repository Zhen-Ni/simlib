"""source.py

This module contains the basic sources in control system.
"""

import math
from ..simsys import BaseBlock

__all__ = ['Clock', 'SineWave', 'Impulse', 'Step', 'Constant',
           'RepeatingSequence', 'UserDefinedSource', 'GaussianNoise']


class Clock(BaseBlock):
    """Block of clock which provides time as single output.

    Parameters
    ----------
    dt: float, optional
        Sampling time.

    name: string, optional
        Name of this block. (default = 'Clock')

    Ports
    -----
    Out[0]: Time
        Current time of the analysis.
    """

    def __init__(self, dt=None, name='Clock'):
        super().__init__(nin=0, nout=1, dt=dt, name=name)

    def BLOCKSTEP(self, *xs):
        return self.t,


class SineWave(BaseBlock):

    """Block of sineWave source.

    Signal source for generating sinewave x = A*sin(w*t+varphi).

    Parameters
    ----------
    amplitude: float, optional
        Amplitude of sinewave. (default = 1.0)

    frequency: float, optional
        Freuency in Hz. (default = 1.0)

    phase: float, optional
        Initial phase in rad/s. (default = 0.0)

    dt: float, optional
        Sampling time.

    name: string, optional
        Name of this block. (default = 'SineWave')

    See Also
    ----------
    BaseBlock
    """

    def __init__(self, amplitude=1.0, frequency=1.0, phase=0.0,
                 dt=None, name='SineWave'):
        super().__init__(nin=0, nout=1, dt=dt, name=name)
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase

    def BLOCKSTEP(self, *xs):
        t = self.t
        y = self.amplitude * \
            math.sin(2 * math.pi * self.frequency * t + self.phase)
        return y,


class Impulse(BaseBlock):

    """Block of impulse signal source.

    Signal source for generating impulse signal x = A * delta(n - n0).

    Parameters
    ----------
    n0: int, optional
        Delay. (default = 0)

    amplitude: float, optional
        Amplitude. (default = 1.0)

    dt: float, optional
        Sampling time.

    name: string, optional
        Name of this block. (default = 'Impulse')

    See Also
    ----------
    BaseBlock
    """

    def __init__(self, n0=0, amplitude=1.0, dt=None, name='Impulse'):
        super().__init__(nin=0, nout=1, dt=dt, name=name)
        self.n0 = n0
        self.amplitude = amplitude

    def BLOCKSTEP(self, *xs):
        y = 0.0
        if self.n == self.n0:
            y = self.amplitude
        return y,


class Step(BaseBlock):

    """Block of step signal source.

    Signal source for generating step signal.
    x = (initial value) if (t < t0) else (final value)

    Parameters
    ----------
    t0: float, optional
        step time. (default = 1.0)

    initial: float, optional
        Initial value. (default = 0.0)

    final: float, optional
        Initial value. (default = 1.0)

    dt: float, optional
        Sampling time.

    name: string, optional
        Name of this block. (default = 'Step')

    See Also
    ----------
    BaseBlock
    """

    def __init__(self, t0=1.0, initial=0.0, final=1.0, dt=None,
                 name='Step'):
        super().__init__(nin=0, nout=1, dt=dt, name=name)
        self.t0 = t0
        self.initial = initial
        self.final = final

    def BLOCKSTEP(self, *xs):
        if self.t < self.t0:
            y = self.initial
        else:
            y = self.final
        return y,


class Constant(BaseBlock):

    """Block of a signal with constant value.

    Parameters
    ----------
    value: float, optional
        The constant value. (default = 1.0)

    dt: float, optional
        Sampling time.

    name: string, optional
        Name of this block. (default = 'Constant')

    See Also
    ----------
    BaseBlock
    """

    def __init__(self, value=1.0, dt=None, name='Constant'):
        super().__init__(nin=0, nout=1, dt=dt, name=name)
        self.value = value

    def BLOCKSTEP(self, *xs):
        y = self.value
        return y,


class RepeatingSequence(BaseBlock):

    """Block of a signal with Repeating Sequence.

    Values in each sequence are interpolated from given time and output.

    Parameters
    ----------
    time: array-like object, optional
        An 1-D array of time values. (default = [0, 1])

    output: array-like object, optional
        An 1-D array of output values. (default = [0, 1])

    dt: float, optional
        Sampling time.

    name: string, optional
        Name of this block. (default = 'Repeating Sequence')

    **kwargs: keyword arguments
        Additonal keyword arguments are passed to scipy.interpolate.interp1d.

    See Also
    ----------
    BaseBlock
    """

    def __init__(self, time=[0, 1], output=[0, 1], dt=None,
                 name='Repeating Sequence', **kwargs):
        """kwargs are passed to scipy.interpolate.interp1d"""
        super().__init__(nin=0, nout=1, dt=dt, name=name)
        self._time = time
        self._output = output
        self._kwargs = kwargs
        from scipy import interpolate
        self._f = interpolate.interp1d(self._time, self._output,
                                       **self._kwargs)

    def BLOCKSTEP(self, *xs):
        y = self._f(self.t % max(self._time))
        return y,


class UserDefinedSource(BaseBlock):

    """Block of a user defined signal source.

    A user-defined function with time as parameter and a number as output
    can be used here for generating signals.

    Parameters
    ----------
    func: function
        A python function generating signal. The function takes one argument
        as time and returns the value of signal at this time.

    dt: float, optional
        Sampling time.

    name: string, optional
        Name of this block. (default = 'User Defined Source')

    See Also
    ----------
    BaseBlock
    """

    def __init__(self, func, dt=None, name='User Defined Source'):
        super().__init__(nin=0, nout=1, dt=dt, name=name)
        self._func = func

    def BLOCKSTEP(self):
        t = self.t
        return self._func(t),


class GaussianNoise(BaseBlock):

    """Block of gaussian white noise source.

    The gaussian noise source generates normally distributed white noise with
    given variance and mean.

    Parameters
    ----------
    variance: float, optional
        Variance of generated noise signal. (default = 1.0)

    mean: float, optional
        Mean value of generated noise signal. (default = 0.0)

    seed: int or array-like object, optional
        Seed for numpy.random.RandomState.

    dt: float, optional
        Sampling time.

    name: string, optional
        Name of this block. (default = 'Gaussian Noise')

    See Also
    ----------
    BaseBlock
    """

    def __init__(self, variance=1.0, mean=0.0, seed=None, dt=None,
                 name='Gaussian Noise'):
        super().__init__(nin=0, nout=1, dt=dt, name=name)
        self._sigma = variance ** 0.5
        self._mean = mean
        self._seed = seed

    def INITFUNC(self):
        from random import gauss, seed
        seed(self._seed)
        # note that random.gauss is NOT thread-safe without a lock around calls
        self._gauss = gauss

    def BLOCKSTEP(self, *xs):
        y = self._gauss(self._mean, self._sigma)
        return y,
