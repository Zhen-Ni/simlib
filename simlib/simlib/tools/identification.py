#!/usr/bin/env python3

import numpy as np
from numpy.linalg import solve
import scipy.signal

from ..misc import as_uint


__all__ = ['identification_frequency_domain_continuous',
           'identification_frequency_domain_discrete']


def identification_time_domain_discrete(u, y, m, n, dt=1, weight=None):
    """Use the Least Squares method to identify a parametric model.

    The system can be written as:
        G(z) = ((a[0]*z^0 + a[1]*z^{-1} + ... + a[M]*z[-M]) /
                (1 + b[0]*z^{-1} + ... + b[N]*z[-N-1]))
    This function calctlates the cofficients of the system.

    Parameters
    ----------
    u: array-like object
        Input signal of a system.

    y: array-like object
        Output signal of a system.

    m: int
        Order of numerator.

    n: int
        Order of denominator.

    dt: float
        Sampling frequency. (default = 1)

    weight: {'input', 'output', 'frequency response'}, optional
        Weight function. Use the spectrum of input, output or frequency
        response as weight function.

    Returns
    -------
    ds: scipy.signal.TransferFunction
        The discrete system.

    Notes
    -----
    The form of the expression of discrete system is different from that of the
    continuous one.
    """

    # Can be ported from my nbserver:Signal Processing/identification.py
    assert 0, 'Not implemented'

    return


def identification_frequency_domain_discrete(freq, h, dt, m, n, weight='h'):
    """Use the Frequency Response Least Squares method to identify a parametric
    model.

    The system can be written as:
        G(z) = ((a[0]*z^0 + a[1]*z^{-1} + ... + a[M]*z[-M]) /
                (1 + b[0]*z^{-1} + ... + b[N]*z[-N-1]))
    This function calctlates the cofficients of the system.

    Parameters
    ----------
    freq: array-like object
        Input frequencies. (Hz)

    h: array-like object
        Frequency response.

    dt: float
        Sampling frequency.

    m: int
        Order of numerator.

    n: int
        Order of denominator.

    weight: array-like object, optional
        Weight vector.

    Returns
    -------
    ds: scipy.signal.TransferFunction
        The discrete system.

    Notes
    -----
    The form of the expression of discrete system is different from that of the
    continuous one.
    """
    m = as_uint(m, False, msg='m should be non-negative int')
    n = as_uint(n, False, msg='n should be non-negative int')
    if not len(freq) == len(h):
        raise ValueError("freq and h must have the same length.")
    if 0.5 / max(freq) < dt:
        import sys
        sys.stderr.write("maximum frequency above nyquist frequency.\n")

    N = len(freq)
    FH = np.array(h)

    # assemble varPhi
    varPhi = np.zeros([N * 2, m + n + 1])
    # assemble the left side of varPhi
    for i in range(n):
        column_i = np.exp(-(i + 1) * 1j * 2 * np.pi * dt * freq) * FH
        varPhi[:N, i] = column_i.real
        varPhi[N:, i] = column_i.imag
    # assemble the right side of varPhi
    for i in range(m + 1):
        column_i = np.exp(-i * 1j * 2 * np.pi * dt * freq)
        varPhi[:N, n + i] = -column_i.real
        varPhi[N:, n + i] = -column_i.imag

    # assemble y
    y = np.zeros([N * 2])
    y[:N] = -FH.real
    y[N:] = -FH.imag

    # 权重矩阵
    if weight == 'h' or weight == 'H':
        weights = abs(FH)**2
        W = np.diag(weights.repeat(2))
    elif weight == 'i' or weight == 'I':
        W = np.eye(len(FH) * 2)  # 单位权重（即不加权）
    else:
        W = weight

    # 求解
    theta = solve(varPhi.T.dot(W).dot(varPhi), varPhi.T.dot(W).dot(y))
    a = theta[:n]
    b = theta[n:]

    a = np.concatenate([[1], a])
    if len(a) < len(b):
        delta = len(b) - len(a)
        num = b
        den = np.concatenate([a, np.zeros(delta)])
    elif len(a) == len(b):
        num = b
        den = a
    else:
        delta = len(a) - len(b)
        num = np.concatenate([b, np.zeros(delta)])
        den = a
    system = scipy.signal.TransferFunction(num, den, dt=dt)
    return system


def identification_frequency_domain_continuous(freq, h, m, n, weight='h'):
    """Use the Frequency Response Least Squares method to identify a parametric
    model.

    The system can be written as:
        H(s) = (b0*s^m + ... + b(m)*s^0) / (a0*s^n + ... + a(n-1)*s + 1)
    This function calctlate the cofficients of the system.

    The details of the algorithm can be found in Ref[1]

    Parameters
    ----------
    freq: array-like object
        Input frequencies (Hz).

    h: array-like object
        Frequency response.

    m: int
        Order of numerator.

    n: int
        Order of denominator.

    weight: float, optional
        Weight function. (default = 'h')

    Returns
    -------
    system: scipy.signal.TransferFunction
        The continuous system

    References
    ----------
    [1] Isermann R, Münchhof M. Identification of Dynamic Systems[M]. Springer
    Berlin Heidelberg, 2011.
    """
    # 见《动态系统辨识——导论与应用》P274
    m = as_uint(m, False, msg='m should be non-negative int')
    n = as_uint(n, False, msg='n should be non-negative int')
    if m > n:
        raise ValueError("m should not be greater than n.")
    if not len(freq) == len(h):
        raise ValueError("freq and h must have the same length.")
    Hjw = np.array(h)

    Rjw = Hjw.real
    Ijw = Hjw.imag
    varPhi = np.zeros([len(Hjw) * 2, m + n + 1])
    y = np.zeros([len(Hjw) * 2])
    for i in range(len(Hjw)):
        w = freq[i] * 2 * np.pi
        R, I = Rjw[i], Ijw[i]
        # varPhi左半边的系数
        varPhi[2 * i, :n] = ([I, R] * (n // 2 + 1))[:n]
        varPhi[2 * i + 1, :n] = ([R, I] * (n // 2 + 1))[:n]
        varPhi[2 * i, :n] *= [((1j)**i).real +
                              ((1j)**i).imag for i in range(2, n + 2)]
        varPhi[2 * i + 1, :n] *= [((1j)**i).real +
                                  ((1j)**i).imag for i in range(1, n + 1)]
        varPhi[2 * i, :n] *= [w**i for i in range(1, n + 1)]
        varPhi[2 * i + 1, :n] *= [w**i for i in range(1, n + 1)]
        # varPhi右半边的系数
        varPhi[2 * i, n:] = [-1 *
                             (1j**i).real * w**i for i in range(m + 1)]
        varPhi[2 * i + 1, n:] = [-1 *
                                 (1j**i).imag * w**i for i in range(m + 1)]
        # y
        y[2 * i:2 * i + 2] = R, I
    # 权重矩阵
    if weight == 'h' or weight == 'H':
        weights = abs(Hjw)**2
        W = np.diag(weights.repeat(2))
    elif weight == 'i' or weight == 'I':
        W = np.eye(len(Hjw) * 2)  # 单位权重（即不加权）
    else:
        W = weight
    # 求解
    theta = solve(varPhi.T.dot(W).dot(varPhi), varPhi.T.dot(W).dot(-y))
    a = theta[:n]
    b = theta[n:]
    system = scipy.signal.TransferFunction(b[::-1],
                                           np.concatenate([a[::-1], [1]]))
    return system
