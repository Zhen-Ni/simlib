"""tools.py

Provide some useful tools for simlib.
"""

import numpy as np
import scipy.signal

__all__ = ['sample_system', 'get_minimum_phase_system_continuous']


def sample_system(tf, dt):
    """Sample a continuous system.

    Parameters
    ----------
    tf: scipy.signal.dlti
        Should be a continuous system.

    dt: float
        Sampling time of the system.

    Output
    ------
    ds: scipy.signal.TransferFunction
        A sampled system.
    """
    tf = scipy.signal.TransferFunction(tf)
    poles = np.exp(tf.poles * dt)
    zeros = np.exp(tf.zeros * dt)
    gain = tf.num[-1] / tf.den[-1]
    den = np.poly(poles)
    num = np.poly(zeros)
    k = gain / (np.sum(num) / np.sum(den))
    ds = scipy.signal.TransferFunction(num * k, den, dt=dt)
    return ds


def get_minimum_phase_system_continuous(system, freq_max=None, sigma_min=None,
                                        keep_gain=True):
    """Get the minimum phase system part from the given continuous system.

    This function works by directly delete the zeros and poles on the right
    side of the  imaginary axis. Optional parameter freq_max provides the
    function to truncate the zeros and poles above the given frequency.
    Optional parameter sigma_min provides the function to truncate the zeros
    and poles below the given real part.

    Parameters
    ----------
    system: scipy.signal.dlti
        Should be a continuous system.

    freq_max: float, optional
        The maximun frequency of the poles and zeros, should be positive.

    sigma_min: float, optional
        The minimum real part of the poles and zeros, should be negative.

    keep_gain: bool, optional
        Whether to keep the static gain of the resultant system. (default =
        True)

    Output
    ------
    system_stablized: scipy.signal.dlti
        A minimum_phase system.

    """
    system = system.to_zpk()
    z, p, k = system.zeros, system.poles, system.gain
    # 频率截断
    if freq_max is not None:
        z = z[abs(z.imag) / 2 / np.pi < freq_max]
        p = p[abs(p.imag) / 2 / np.pi < freq_max]
    # 去掉实部过小的零极点
    if sigma_min is not None:
        p = p[p.real > sigma_min]
        z = z[z.real > sigma_min]
    # 去掉不稳定的零极点（保证系统的最小相位特性）
    # 实际上是因为如果只去掉虚轴右侧极点的话，会导致辨识得到的系统不准
    p = p[p.real < 0]
    z = z[z.real < 0]
    # 调整静态增益
    system_stablized = scipy.signal.ZerosPolesGain(z, p, k)
    if keep_gain:
        w, hs = scipy.signal.freqresp(system_stablized, [0])
        w, ho = scipy.signal.freqresp(system, [0])
        new_gain = system_stablized.gain / hs[0] * ho[0]
        system_stablized.gain = new_gain
    return system_stablized