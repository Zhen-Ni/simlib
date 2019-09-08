"""tools.py

Provide some useful tools for simlib.
"""

import numpy as np
import scipy.signal

__all__ = ['sample_system', 'get_minimum_phase_system',
           'get_minimum_phase_system_continuous',
           'get_minimum_phase_system_discrete',
           'modal_decomposition']


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


def get_minimum_phase_system(system, *args, **kwargs):
    """Get the minimum phase system part from the given system.
    See 'get_minimum_phase_system_continuous' and
    'get_minimum_phase_system_discrete' for more information.
    """
    if isinstance(system, scipy.signal.lti):
        return get_minimum_phase_system_continuous(system, *args, **kwargs)
    elif isinstance(system, scipy.signal.dlti):
        return get_minimum_phase_system_discrete(system, *args, **kwargs)
    else:
        raise ValueError('`system` must be lti or dlti object')


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
    system: scipy.signal.lti
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
    system_stablized: scipy.signal.lti
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


def get_minimum_phase_system_discrete(system, abs_min=None, keep_gain=True):
    """Get the minimum phase system part from the given discrete system.

    This function works by directly delete the zeros and poles outside the unit
    circle. Optional parameter abs_min provides the function to truncate the
    zeros and poles inside the circle with given radius.

    Parameters
    ----------
    system: scipy.signal.dlti
        Should be a discrete and casual system.

    abs_min: float, optional
        The minimum absolute value of the poles and zeros, should be positive
        and smaller than 1.

    keep_gain: bool, optional
        Whether to keep the static gain of the resultant system. (default =
        True)

    Output
    ------
    system_stablized: scipy.signal.dlti
        A minimum_phase system.

    """
    system = system.to_zpk()
    z, p, k, dt = system.zeros, system.poles, system.gain, system.dt
    # 离散系统，不需要进行频率截断
    # 去掉实部过小的零极点
    if abs_min is not None:
        p = p[abs(p) > abs_min]
        z = z[abs(z) > abs_min]
    # 去掉不稳定的零极点（保证系统的最小相位特性）
    # 实际上是因为如果只去掉虚轴右侧极点的话，会导致辨识得到的系统不准
    else:
        p = p[abs(p) < 1]
        z = z[abs(z) < 1]
    # 调整静态增益
    system_stablized = scipy.signal.ZerosPolesGain(z, p, k, dt=dt)
    if keep_gain:
        w, hs = scipy.signal.dfreqresp(system_stablized, [0])
        w, ho = scipy.signal.dfreqresp(system, [0])
        new_gain = system_stablized.gain / hs[0] * ho[0]
        system_stablized.gain = new_gain
    return system_stablized


def modal_decomposition(system):
    """Decomposite a continuous system into modal space.

    Parameters
    ----------
    system : scipy.signal.lti
        The continuous system

    Returns
    -------
    k: ndarray
        Modal participation factor.

    xi: ndarray
        Modal damping.

    wn: ndarray
        Modal frequency in rad/s

    residue: scipy.signal.TransferFunction or None
        Portion of the system which can not be decomposited into modal domain.

    See Also
    --------
    modal_superposition
    """
    system = system.to_tf()
    num, den = system.num, system.den
    if num.imag.any() or den.imag.any():
        raise ValueError('system must be real')
    # set tol=0.0 to make sure there are no repeated roots
    r, p, k_ = scipy.signal.residue(system.num, system.den, tol=0.0)
    r1, p1 = [], []    # 一阶子系统
    r2, p2 = [], []    # 二阶子系统
    for i in range(len(r)):
        if p[i].imag:
            # 出现共轭复数根的情况下，只需要储存其中的一个就可以了
            if p[i].imag > 0:
                r2.append(r[i])
                p2.append(p[i])
        else:
            r1.append(r[i])
            p1.append(p[i])

    r, p = [], []
    k, xi, wn = [], [], []
    # 一阶子系统无法通过模态坐标表示
    r.extend(r1)
    p.extend(p1)
    # 二阶子系统的r和p均为共轭复数对，表示为r=rr+1j*ri, p=pr+1j*pi
    for i in range(len(r2)):
        rr, ri = r2[i].real, r2[i].imag
        pr, pi = p2[i].real, p2[i].imag
        p_1 = pr + 1j * pi
        p_2 = pr - 1j * pi
        # 分子含s的项无法通过模态坐标表示
        if rr:    # in case the system doesn't have parts containing the s term
            a = rr * p_1 / (1j * pi)
            b = -rr * p_2 / (1j * pi)
            r.append(a)
            r.append(b)
            p.append(p_1)
            p.append(p_2)
        # 剩余部分用模态坐标表示
        w = (pr**2 + pi**2)**0.5
        k.append(-2 * rr * pr - 2 * ri * pi)
        xi.append(-pr / w)
        wn.append(w)

    k = np.array(k)
    xi = np.array(xi)
    wn = np.array(wn)

    if len(r):
        residue = scipy.signal.invres(r, p, k_, tol=0)
        residue = scipy.signal.TransferFunction(*residue)
    elif k_.any():
        residue = scipy.signal.TransferFunction(k_, [1])
    else:
        residue = None
    return k, xi, wn, residue


def modal_superposition(k, xi, wn, residue=None, tol=1e-3):
    """Construct a continuous system using parameters in modal space.

    Note that the system is assumed to have real parameters.

    Parameters
    -------
    k: ndarray
        Modal participation factor.

    xi: ndarray
        Modal damping.

    wn: ndarray
        Modal frequency in rad/s.

    residue: scipy.signal.TransferFunction or None
        Portion of the system which can not be decomposited into modal domain.
        (defaults to None)

    tol: float, optional
        The tolerance for two roots to be considered equal. Default is 1e-6.

    Returns
    ----------
    system : scipy.signal.lti
        The continuous system

    See also
    --------
    modal_decomposition
    """
    n = len(k)
    if not (n == len(xi) or n == len(wn)):
        raise ValueError('k, xi and wn should have the same length')

    if residue is None:
        num = np.array([0])
        den = np.array([1])
    else:
        num = np.array(residue.num)
        den = np.array(residue.den)

    for i in range(n):
        numi = np.array([k[i]])
        deni = np.array([1, 2 * xi[i] * wn[i], wn[i] * wn[i]])
        num = np.polyadd(np.polymul(num, deni), np.polymul(numi, den))
        den = np.polymul(den, deni)

    # 多项式化简
    poles = np.sort(np.roots(den))
    zeros = np.sort(np.roots(num))

    common = []
    it_poles = 0
    it_zeros = 0
    while it_zeros != len(zeros) and it_poles != len(poles):
        p = poles[it_poles]
        z = zeros[it_zeros]
        if _complex_equal(p, z, tol):
            common.append(p)
            it_zeros += 1
            it_poles += 1
        elif _complex_compare(z, p, tol):
            it_zeros += 1
        else:
            it_poles += 1

    divisor = np.poly(common)
    den = np.polydiv(den, divisor)[0].real
    num = np.polydiv(num, divisor)[0].real

    return scipy.signal.TransferFunction(num, den)


def _complex_equal(a, b, tol):
    if abs(a.real - b.real) < tol:
        if abs(a.imag - b.imag) < tol:
            return True
    return False


def _complex_compare(a, b, tol):
    if abs(a.real - b.real) < tol:
        if abs(a.imag - b.imag) < tol:
            return False
        elif a.imag < b.imag:
            return True
        else:
            return False
    elif a.real < b.real:
        return True
    else:
        return False


if __name__ == '__main__':
    import scipy.signal as signal
    NUM_C = 5.078e-7, 6.602e-6, 0.05012
    DEN_C = 1.055e-9, 2.109e-8, 1.82e-4, 9.7e-4, 0.9
    SYS_C = signal.TransferFunction(NUM_C, DEN_C)
    T = 0.001
    SYS_D = sample_system(SYS_C, T)
    SYS_DI = signal.TransferFunction(SYS_D.den, SYS_D.num, dt=T)
