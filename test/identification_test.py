#!/usr/bin/env python3

import os
import sys
import scipy.signal as signal
from pylab import *

os.chdir('..')
sys.path.append(os.curdir)

from simlib import *


num = array([-6.22555550e-02, -2.45048595e+00, -2.18491996e+05, -3.99817130e+06,
        1.08054320e+11,  9.80417464e+11,  1.91582239e+17,  5.18081582e+17,
        2.67653807e+22,  1.39287712e+22,  3.40261839e+26])
den = array([1.00000000e+00, 2.08130860e+01, 2.54906567e+06, 3.00062849e+07,
       1.60888134e+12, 1.04231827e+13, 3.22061398e+17, 1.18076901e+18,
       2.21969856e+22, 3.51124582e+22, 3.04184198e+26])
FS = 400

def test_identification_frequency_domain_continuous():
    m = 12
    n = 12

    system0 = signal.TransferFunction(num,den)
    freq = linspace(0,200,801)
    freq[0] = 1e-3
    _freq, mag, phase = signal.bode((system0.num,system0.den), freq*2*pi)
    mag0 = 10**(mag/20)
    phase0 = phase / 180 * pi

    system1 = identification_frequency_domain_continuous(freq, mag0*exp(1j*phase0) ,m ,n, weight='h')
    _freq, mag, phase = signal.bode(system1, freq*2*pi)
    mag1 = 10**(mag/20)
    phase1 = phase / 180 * pi

    figure()
    semilogy(freq, mag0,lw=5)
    semilogy(freq, mag1)
    grid()
    figure()
    plot(freq, phase0,lw=5)
    plot(freq, phase1)
    grid()
    return system1

def test_identification_frequency_domain_discrete():
    m = 12
    n = 12

    system0 = signal.TransferFunction(num,den)
    freq = linspace(0,200,801)
    freq[0] = 1e-3
    _freq, mag, phase = signal.bode((system0.num,system0.den), freq*2*pi)
    mag0 = 10**(mag/20)
    phase0 = phase / 180 * pi

    system1 = identification_frequency_domain_discrete(freq, mag0*exp(1j*phase0), 1/FS,m ,n, weight='h')
    _freq, mag, phase = signal.dbode(system1,freq/FS*2*pi)
    mag1 = 10**(mag/20)
    phase1 = phase/180*pi

    figure()
    semilogy(freq, mag0,lw=5)
    semilogy(freq, mag1)
    grid()
    figure()
    plot(freq, phase0,lw=5)
    plot(freq, phase1)
    grid()
    return system1

if __name__ == '__main__':
    test_identification_frequency_domain_continuous()
    test_identification_frequency_domain_discrete()
