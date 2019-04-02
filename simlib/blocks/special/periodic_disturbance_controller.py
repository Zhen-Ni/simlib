"""periodic_disturbance_controller.py

A collection of control algorithms for adaptive control.
"""


import numpy as np
import scipy.signal as signal

from ...simsys import BaseBlock

__all__ = ['PDCClassic', 'PDCImproved', 'PDC']


class PDCClassic(BaseBlock):
    """Periodic disturbance controller.

    This controller uses the adaptive feed-forward technique to control several
    periodic disturbances with known frequencies.

    Parameters
    ----------
    w: iterable
        The circular freuencies of the disturbance.

    func_response: callable
        This function takes `w` as the only parameter and returns the estimated
        frequency response of the plant.

    mu: float
        The adaptive gain.

    dt: float, optional
        Sampling time. If not given, default to the same as sampling time of
        the system.

    name: string, optional
        Name of this block. (default to 'PDC_classic')

    Ports
    -----
    In[0]: Error
        The error signal.

    Out[0]: Output
        The output of the controller.

    Reference
    ---------
    My paper on ICSV26.
    """

    def __init__(self, w, func_response, mu, dt=None, name='PDCClassic'):
        super().__init__(nin=1, nout=1, name=name)
        self.inports[0].rename('Error')
        self.outports[0].rename('Output')
        self._w = np.array(w)
        self._func_response = func_response
        self._mu = np.array(mu)
        # theta_c and theta_s are the amplitudes of the cosine and sine
        # components of the disturbance
        self._theta_c = None
        self._theta_s = None
        self._g = None

    @property
    def set_w(self, w):
        """Set value for circular frequency w."""
        self._w = np.array(w)

    @property
    def w(self):
        """Get the circular frequency."""
        return self._w

    @property
    def components(self):
        """Get the cosine and sine components of the controller."""
        return self._theta_c, self._theta_s

    def INITFUNC(self):
        n = len(self._w)
        self._theta_c = np.zeros(n)
        self._theta_s = np.zeros(n)
        self._g = self._func_response(self._w)

    def BLOCKSTEP(self, *xs):
        err = xs[0]
        t = self.t
        g = self._g
        mu = self._mu
        wt = self._w * t
        c = np.cos(wt)
        s = np.sin(wt)
        self._theta_c -= 2 * err * mu * (g.real * c - g.imag * s)
        self._theta_s -= 2 * err * mu * (g.real * s + g.imag * c)
        return np.sum(self._theta_c * c + self._theta_s * s),


class _PDCBase(BaseBlock):
    """ Base class for periodic disturbance controller with frequency tracker
    and faster convergence.

    This controller uses the adaptive feed-forward technique to control several
    periodic disturbances with inaccurate frequencies. The frequency tracker
    is able to track the accurate frequency if an appropriate initial value is
    given. Metric conversion is also used to accelerate the convergence.

    Parameters
    ----------
    n_components: int
        Number of frequency components in disturbance.

    func_response: callable
        This function takes `w0` as the only parameter and returns the
        estimated frequency response of the plant.

    mu_global: float
        The adaptive gain for the whole iteration procedure.

    mu_omega: float, optional
        The additional multiplier for adaptive gain for frequency estimator.
        (default to 1)

    dt: float, optional
        Sampling time. If not given, default to the same as sampling time of
        the system.

    name: string, optional
        Name of this block. (default to 'PDC_improved')

    Ports
    -----
    In[0]: Error
        The error signal.

    Out[0]: Output
        The output of the controller.

    Notes
    -----
    Directly using this block will not produce correct results for controlling
    periodic disturbance as the initial values for the frequencies are all
    zeros in this class. There are two ways to set the initial frequencies. The
    first is to use PDCImproved block, which an esitimation of the frequencies
    should be defined by the user. Another solution is to estimate the
    frequencies by performing an spectral analysis, and users may refer to PDC
    block if they would like to use this method.

    See Also
    ---------
    PDCImproved, PDC
    """

    def __init__(self, n_components, func_response, mu_global, mu_omega=1,
                 dt=None, name='_PDCBase'):
        super().__init__(nin=1, nout=1, name=name)
        self._n_components = n_components
        self._func_response = func_response
        self._mu_global = np.array(mu_global)
        self._mu_omega = np.array(mu_omega)
        # A and B are the amplitudes of the cosine and sine components of the
        # disturbance, self._phase records the current phase, and w is the
        # estimation of the frequency
        self._A = None
        self._B = None
        self._phase = None
        self._w = None

    @property
    def w(self):
        """Get the circular frequency."""
        return self._w

    @property
    def components(self):
        """Get the cosine and sine components of the controller."""
        return self._A, self._B

    @property
    def phase(self):
        """Get the current phase."""
        return self._phase

    def INITFUNC(self):
        n = self._n_components
        self._A = np.zeros(n)
        self._B = np.zeros(n)
        self._phase = np.zeros(n)
        # setting self._w to 0 is erroneous, and should be correctly set by
        # derived class
        self._w = np.zeros(n)

    def BLOCKSTEP(self, *xs):
        err = xs[0]
        A = self._A.copy()
        B = self._B.copy()
        g = self._func_response(self._w)
        s = np.sin(self._phase)
        c = np.cos(self._phase)
        self._A -= 2 * err * self._mu_global * (g.real * c - g.imag * s)\
            * self.dt / abs(g)**2
        self._B -= 2 * err * self._mu_global * (g.real * s + g.imag * c)\
            * self.dt / abs(g)**2
        self._w -= 2 * err * self._mu_global * self._mu_omega *\
            (g.real * (-A * s + B * c) - g.imag * (A * c + B * s)) * self.dt /\
            abs(g)**2 / (A**2 + B**2 + 1)

        self._phase += self._w * self.dt
        self._phase[self._phase > 2 * np.pi] -= np.pi * 2
        return np.sum(self._A * np.cos(self._phase) +
                      self._B * np.sin(self._phase)),


class PDCImproved(_PDCBase):
    """Periodic disturbance controller with frequency tracker and faster
    convergence.

    This controller uses the adaptive feed-forward technique to control several
    periodic disturbances with inaccurate frequencies. The frequency tracker
    is able to track the accurate frequency if an appropriate initial value is
    given. Metric conversion is also used to accelerate the convergence.

    Parameters
    ----------
    w0: iterable
        The circular freuencies of the disturbance.

    func_response: callable
        This function takes `w0` as the only parameter and returns the
        estimated frequency response of the plant.

    mu_global: float
        The adaptive gain for the whole iteration procedure.

    mu_omega: float, optional
        The additional multiplier for adaptive gain for frequency estimator.
        (default to 1)

    dt: float, optional
        Sampling time. If not given, default to the same as sampling time of
        the system.

    name: string, optional
        Name of this block. (default to 'PDC_improved')

    Ports
    -----
    In[0]: Error
        The error signal.

    Out[0]: Output
        The output of the controller.

    Reference
    ---------
    My paper on ICSV26.
    """

    def __init__(self, w0, func_response, mu_global, mu_omega=1, dt=None,
                 name='PDC_improved'):
        super().__init__(len(w0), func_response, mu_global, mu_omega, dt, name)
        self._w0 = np.asarray(w0)

    @property
    def w0(self):
        return self._w0

    def set_w0(self, w0):
        """Set initial value for w.

        Note that this function should be used before initialization of the
        system, otherwise, it would have no effect for the current run.
        """
        self._w0 = np.array(w0)
        self._n_components = len(self._w0)

    def INITFUNC(self):
        super().INITFUNC()
        self._w[:] = self._w0


class PDC(_PDCBase):
    """Periodic disturbance controller with frequency estimator, frequency
    tracker and faster convergence.

    This controller uses the adaptive feed-forward technique to control several
    periodic disturbances with inaccurate frequencies. A FFT analysis is
    performed first to get the frequencies of the disturbance. The frequencies
    by the FFT is then feeded to the frequency tracker as its initial value.
    The frequency tracker will track the slow variation of the frequencies.
    Metric conversion is also used to accelerate the convergence.

    Parameters
    ----------
    n_components: iterable
        Number of frequency components in disturbance.

    func_response: callable
        This function takes `w0` as the only parameter and returns the
        estimated frequency response of the plant.

    mu_global: float
        The adaptive gain for the whole iteration procedure.

    mu_omega: float, optional
        The additional multiplier for adaptive gain for frequency estimator.
        (default to 1)

    t_fft: float, optional
        The time for collecting data for fft analysis. The control will not
        begin until simulation time reaches t_fft. (default to 5)

    t_fft_start: float, optional
        The beginning time for gathering data for fft analysis. (default to 0)

    resolution: float or None, optional
        The computational resoltion for fft analysis. If None is provided, the
        computational resolution is 1/t_fft. If a float is given, the
        computational resolution will be set the smallest value 2**n (n is an
        integer) larger than the given value (in Hz) by padding with zeros or
        cropping the input data. (default to None)

    window: callable, optional
        The window used for fft analysis. It should take an integer indicating
        the length of data as input, and returns an iterable object of the
        given length indicating the window. (default to signal.blackman)

    dt: float, optional
        Sampling time. If not given, default to the same as sampling time of
        the system.

    name: string, optional
        Name of this block. (default to 'PDC')

    Ports
    -----
    In[0]: Error
        The error signal.

    Out[0]: Output
        The output of the controller.

    Reference
    ---------
    My paper on ICSV26.
    """

    def __init__(self, n_components, func_response, mu_global, mu_omega=1,
                 t_fft=5, t_fft_start=0, resolution=None, window=signal.blackman,
                 dt=None, name='PDC'):
        super().__init__(n_components, func_response=func_response,
                         mu_global=mu_global, mu_omega=mu_omega, dt=dt, name=name)
        self._t_fft = t_fft
        self._t_fft_start = t_fft_start
        self._resolution = resolution
        self._window = window
        self._data_for_fft = None
        self._fft_finished = None

    @property
    def n_components(self):
        return self._n_components

    def set_n_components(self, n_components):
        """Set initial value for n_components.

        Note that this function should be used before initialization of the
        system, otherwise, it would have no effect for the current run.
        """
        self._n_components = n_components

    def INITFUNC(self):
        super().INITFUNC()
        self._data_for_fft = []
        self._fft_finished = False

    def _get_w0_using_fft_data(self):
        x = self._data_for_fft
        resolution = self._resolution
        window = np.asarray(self._window(len(x)))
        if resolution is None:
            n_fft = len(x)
        else:
            n_fft_minimal = (1.0 / self._t_fft) / resolution * len(x)
            n_fft = 1
            while n_fft < n_fft_minimal:
                n_fft <<= 1
        fx = np.fft.rfft(x * window, n_fft) / len(x)
        freq = np.fft.rfftfreq(n_fft, self.dt)
        peaks = signal.find_peaks(abs(fx))[0]
        peaks = sorted(peaks, key=lambda p: abs(fx)[p], reverse=True)
        peak_freqs = freq[peaks]
#        print(peak_freqs[:5])
        w0 = np.zeros(self._n_components)
        w = peak_freqs[:self._n_components] * 2 * np.pi
        w0[:len(w)] = w
        return w0

    def BLOCKSTEP(self, *xs):
        if self.t < self._t_fft+self._t_fft_start:
            if self.t >= self._t_fft_start:
                self._data_for_fft.append(xs[0])
            return 0,

        if not self._fft_finished:
            self._w = self._get_w0_using_fft_data()
            self._fft_finished = True

        return super().BLOCKSTEP(*xs)
