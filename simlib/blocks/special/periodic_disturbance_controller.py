"""periodic_disturbance_controller.py

A collection of control algorithms for adaptive control.
"""


import numpy as np

from ...simsys import BaseBlock

__all__ = ['PDC_classic', 'PDC_improved', ]


class PDC_classic(BaseBlock):
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

    def __init__(self, w, func_response, mu, dt=None, name='PDC_classic'):
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


class PDC_improved(BaseBlock):
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

    def __init__(self, w0, func_response, mu_global, mu_omega=1, dt=None,
                 name='PDC_improved'):
        super().__init__(nin=1, nout=1, name=name)
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
        self._w0 = np.array(w0)

    @property
    def w0(self):
        return self._w0

    def set_w0(self, w0):
        """Set initial value for w.

        Note that this function should be used before initialization of the
        system, otherwise, it would have no effect for the current run.
        """
        self._w0 = np.array(w0)

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
        n = len(self._w0)
        self._A = np.zeros(n)
        self._B = np.zeros(n)
        self._phase = np.zeros(n)
        self._w = self._w0.copy()

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


'''
class PDC(BaseBlock):
    def __init__(self, n_w, sys, mu_global,mu_omega, t_start_control=5,
                 pad_multiple = 10, window = signal.blackman,
                 name='AFC-controller-improved-with-frequency-analysis'):
        super().__init__(nin=1, nout=1, name=name)
        self.sys = sys
        self.mu_global = np.array(mu_global)
        self.mu_omega = np.array(mu_omega)
        self.t_start_control = t_start_control
        self.pad_multiple = pad_multiple
        self.window = window
        n = int(n_w)
        self.n_w = n
        self.A = np.zeros(n)
        self.B = np.zeros(n)

    def INITFUNC(self):
        self._data_for_fa = []
        self.w = None

    def set_w(self, w):
        w = w[:len(self.w)]
        n = len(w)
        w1 = w
        w0 = self.w[:n]
        A0 = self.A[:n]
        B0 = self.B[:n]
        t = self.t
        phi1 = (w0 - w1) * t
        c = np.cos(phi1)
        s = np.sin(phi1)
        A1 = A0 * c + B0 * s
        B1 = B0 * c - A0 * s
        self.w[:n] = w1
        self.A[:n] = A1
        self.B[:n] = B1

    def get_freq(self):
        x = self._data_for_fa
        pad_multiple = self.pad_multiple
        window = self.window(len(x))
        fx = np.fft.fftshift(np.fft.fft(x*window,len(x)*pad_multiple)) / len(x)
        freq = np.fft.fftshift(np.fft.fftfreq(len(x)*pad_multiple,self.dt))
        fx = fx[freq>=0]
        freq = freq[freq>=0]
        peaks = signal.find_peaks(abs(fx))[0]
        peaks = sorted(peaks, key=lambda p:abs(fx)[p], reverse=True)
        peak_freqs = freq[peaks]
#        print(peak_freqs[:5])
        w0 = np.zeros(self.n_w)
        w = peak_freqs[:self.n_w]*2*np.pi
        w0[:len(w)] = w
        return w0


    def BLOCKSTEP(self, *xs):
        if self.t < self.t_start_control:
            self._data_for_fa.append(xs[0])
            return 0,
        else:
            if self.w is None:
                self.w = self.get_freq()
            err = xs[0]
            t = self.t
            theta = self.w * t
            A = self.A.copy()
            B = self.B.copy()
            g = signal.freqresp(self.sys,self.w)[1]
            self.A -= 2 * err *self.mu_global* \
                      (g.real*np.cos(theta)-g.imag*np.sin(theta))\
                      *self.dt/ abs(g)**2
            self.B -= 2 * err *self.mu_global* \
                      (g.real*np.sin(theta)+g.imag*np.cos(theta))\
                      *self.dt/ abs(g)**2
            w = self.w - 2 * err *self.mu_global*self.mu_omega*\
                (g.real*(-A*np.sin(theta)+B*np.cos(theta)) \
                -g.imag*(A*np.cos(theta)+B*np.sin(theta))) *self.dt/\
                 abs(g)**2/(A**2+B**2+1)
            self.set_w(w)
            return np.sum(self.A*np.cos(self.w*t)+self.B*np.sin(self.w*t)),

'''
