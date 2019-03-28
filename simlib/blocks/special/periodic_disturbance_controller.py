"""periodic_disturbance_controller.py

A collection of control algorithms for adaptive control.
"""


import numpy as np

from ...simsys import BaseBlock

__all__ = ['PDC_classic', ]


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

    def __init__(self, w, func_response, mu, name='PDC_classic'):
        super().__init__(nin=1, nout=1, name=name)
        self.inports[0].rename('Error')
        self.outports[0].rename('Output')
        self.w = np.array(w)
        self.g = func_response(self.w)
        self.mu = np.array(mu)
        n = len(w)
        self.theta_c = np.zeros(n)
        self.theta_s = np.zeros(n)

    def BLOCKSTEP(self, *xs):
        err = xs[0]
        t = self.t
        wt = self.w * t
        c = np.cos(wt)
        s = np.sin(wt)
        self.theta_c -= 2 * err * self.mu * (self.g.real * c - self.g.imag * s)
        self.theta_s -= 2 * err * self.mu * (self.g.real * s + self.g.imag * c)
        return np.sum(self.theta_c * c + self.theta_s * s),


'''
class PDC_improved(BaseBlock):
    def __init__(self, w, sys, mu_global,mu_omega, name='AFC-controller-improved'):
        super().__init__(nin=1, nout=1, name=name)
        self.sys = sys
        self.mu_global = np.array(mu_global)
        self.mu_omega = np.array(mu_omega)
        n = len(w)
        self.A = np.zeros(n)
        self.B = np.zeros(n)
        self.w0 = np.array(w)

    def INITFUNC(self):
        self.w = self.w0

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

    def BLOCKSTEP(self, *xs):
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
