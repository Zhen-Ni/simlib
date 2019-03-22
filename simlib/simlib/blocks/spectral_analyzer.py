"""spectral_analyzer.py

A collection of components for spectral analysis.
"""


import copy
import numpy as np
from collections import deque
import scipy.fftpack as fftpack
import scipy.signal as signal

from ..misc import as_uint
from ..simsys import BaseBlock


__all__ = ['FourierTransformer', 'PowerSpectrum']


class FourierTransformer(BaseBlock):
    """Fourier Transformation of the signal.

    The Fourier Transformation can be calculated using a given number of input
    data or using all input data. 

    Parameters
    ----------
    npoints: int
        Number of points to do the FFT. if nponits == 0, all input data will be
        used to do the FFT.

    noverlap: int, optional
        Number of points to overlap between segments. (default = 0)

    nfft: int, optional
        Length of the FFT used. If None the length of npoints will be used.

    fftshift: int, optional
        Whether to hift the zero-frequency component to the center of the
        spectrum. (default = True)

    normalize: bool, optional
        Whether to normalize the result of the fft. (default = False)

    dt: float, optional
        Sampling time.

    name: string, optional
        Name of this block. (default = 'FourierTransform')

    Ports
    -----
    In[0]: Input
        The input signal.

    Out[0]: FFT Frequency
        The frequencies of FFT Result.

    Out[1]: FFT Result
        The output signal array.
    """

    def __init__(self, npoints, noverlap=0, nfft=None, fftshift=True,
                 normalize=False, dt=None, name='FourierTransform'):
        super().__init__(nin=1, nout=2, dt=dt, name=name)
        self.inports[0].rename('Input')
        self.outports[0].rename('FFT Frequency')
        self.outports[1].rename('FFT Result')
        # format parameters
        self._npoints = as_uint(npoints, positive_number=False,
                                msg='npoints must be int >= 0')
        try:
            self._noverlap = int(noverlap)
        except Exception:
            raise ValueError('noverlap must be int')
        if self._noverlap >= self._npoints and self._npoints != 0:
            raise ValueError('noverlap must be less than npoints')
        if nfft is None:
            self._nfft = None
        else:
            self._nfft = as_uint(nfft, msg='npoints must be int > 0')
        self._fftshift = fftshift
        self._normalize = normalize

        self._x_init = deque([0.0] * self._npoints)

    def INITFUNC(self):
        self._x = copy.copy(self._x_init)
        self._fx = np.zeros_like(self._x, dtype=complex)
        self._freq = np.zeros_like(self._x, dtype=float)

    def BLOCKSTEP(self, *xs):
        xn = xs[0]
        self._x.append(xn)
        # if self._npoints == 0, we should store all input data
        if self._npoints:
            self._x.popleft()
        # do fft for certain time periods
        if (self._npoints == 0 or
                (self._n + 1) % (self._npoints - self._noverlap) == 0):
            fx = fftpack.fft(self._x, self._nfft)
            if self._normalize:
                fx /= len(fx)
            freq = fftpack.fftfreq(len(fx), self.dt)
            if self._fftshift:
                fx = fftpack.fftshift(fx)
                freq = fftpack.fftshift(freq)
            self._fx = fx
            self._freq = freq

        return self._freq, self._fx


class PowerSpectrum(BaseBlock):
    """Power spectrum of the signal.

    The Power spectrum can be calculated using a given number of input data or
    using all input data. 

    Parameters
    ----------
    npoints: int
        Number of points to calculate the power spectrum. if nponits == 0, all
        input data will be used to do the calculation.

    noverlap: int, optional
        Number of points to overlap between segments. (default = 0)

    window : str or tuple or array_like, optional
        Desired window to use. If window is a string or tuple, it is passed to
        scipy.signal.get_window to generate the window values, which are
        DFT-even by default. If window is array_like it will be used directly
        as the window and its length must be nperseg. Defaults to ‘boxcar’.

    nfft: int, optional
        Length of the FFT used. If None the length of npoints will be used.
        Defaults to None.

    detrend : str or function or False, optional
        Specifies how to detrend each segment. If detrend is a string, it is
        passed as the type argument to the detrend function in scipy.signal. If
        it is a function, it takes a segment and returns a detrended segment. 
        If detrend is False, no detrending is done. Defaults to 'constant'.

    scaling : { 'density', 'spectrum' }, optional
        Selects between computing the power spectral density ('density') where
        Pxx has units of V**2/Hz and computing the power spectrum ('spectrum')
        where Pxx has units of V**2, if x is measured in V and fs is measured
        in Hz. Defaults to 'density'.

    dt: float, optional
        Sampling time.

    name: string, optional
        Name of this block. (default = 'PowerSpectrum')

    Ports
    -----
    In[0]: Input
        The input signal.

    Out[0]: Frequency
        Array of sample frequencies.

    Out[1]: Result
        Power spectral density or power spectrum of input.

    See Also
    --------
    scipy.signal.periodogram
    """

    def __init__(self, npoints, noverlap=0, window='boxcar', nfft=None,
                 detrend='constant', scaling='density', dt=None,
                 name='PowerSpectrum'):
        super().__init__(nin=1, nout=2, dt=dt, name=name)
        self.inports[0].rename('Input')
        self.outports[0].rename('Frequency')
        self.outports[1].rename('Result')
        # format parameters
        self._npoints = as_uint(npoints, positive_number=False,
                                msg='npoints must be int >= 0')
        try:
            self._noverlap = int(noverlap)
        except Exception:
            raise ValueError('noverlap must be int')
        if self._noverlap >= self._npoints and self._npoints != 0:
            raise ValueError('noverlap must be less than npoints')
        if nfft is None:
            self._nfft = None
        else:
            self._nfft = as_uint(nfft, msg='npoints must be int > 0')
        self._window = window
        self._detrend = detrend
        self._scaling = scaling

        self._x_init = deque([0.0] * self._npoints)

    def INITFUNC(self):
        self._x = copy.copy(self._x_init)
        self._pxx = np.zeros_like(self._x, dtype=float)
        self._freq = np.zeros_like(self._x, dtype=float)

    def BLOCKSTEP(self, *xs):
        xn = xs[0]
        self._x.append(xn)
        # if self._npoints == 0, we should store all input data
        if self._npoints:
            self._x.popleft()
        # do fft at certain time periods
        if (self._npoints == 0 or
                (self._n + 1) % (self._npoints - self._noverlap) == 0):
            freq, pxx = signal.periodogram(self._x, fs=1 / self.dt,
                                           window=self._window,
                                           nfft=self._nfft,
                                           detrend=self._detrend,
                                           scaling=self._scaling)
            self._pxx = pxx
            self._freq = freq

        return self._freq, self._pxx
