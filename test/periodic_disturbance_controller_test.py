#!/usr/bin/env python3

import os
import sys
os.chdir('..')
sys.path.append(os.curdir)

from pylab import *

import simlib as sim

import scipy.signal as signal
NUM_C = 5.078e-7, 6.602e-6, 0.05012
DEN_C = 1.055e-9, 2.109e-8, 1.82e-4, 9.7e-4, 0.9
SYS_C = signal.TransferFunction(NUM_C, DEN_C)


T = 0.001
SYS_D = sim.sample_system(SYS_C, T)

# 等效干扰信号


def func_source(t):
    y1 = 1 * np.sin(2 * np.pi * 7.97 * t)
    y2 = 1 * np.cos(2 * np.pi * 12.05 * t + 13)
    y3 = 1 * np.cos(2 * np.pi * 65 * t)
    d = 0 * np.random.randn()
    return y1 + y2 + y3 + d


#freq_initial = np.array([8,12,65])*2*np.pi
freq_initial = np.array([8, 12, 65]) * 2 * np.pi


def show_bode():
    # Show the bode plot for the targeted system
    import numpy as np
    FREQ_RANGE = np.linspace(0, 100, 501)
    w, mag_c, phase_c = signal.bode(SYS_C, FREQ_RANGE * 2 * np.pi)
    w, mag_d, phase_d = signal.dbode(SYS_D, FREQ_RANGE / (1 / T) * 2 * np.pi)

    fig = figure(figsize=(12, 9))
    ax = fig.add_subplot(211)
    ax.semilogx(FREQ_RANGE, mag_c, lw=5, label='continuous')
    ax.semilogx(FREQ_RANGE, mag_d, label='discrete')
    ax.grid(which='both')
    ax.legend()
    ax.set_ylabel('Magnitude (dB)')
    ax = fig.add_subplot(212, sharex=ax)
    ax.semilogx(FREQ_RANGE, phase_c, lw=5, label='continuous')
    ax.semilogx(FREQ_RANGE, phase_d, label='discrete')
    ax.grid(which='both')
    ax.legend()
    ax.set_ylabel('Phase (deg)')
    ax.set_xlabel('Frequency (Hz)\nNormalized Frequency')
    ax.set_xticklabels(['{freq}\n{Omega}'.format(
        freq=i, Omega=2 * i * T) for i in ax.get_xticks()])
    show()


def test_PDC_classic(mu=1e-2):
    t_control_start = 5

    system = sim.System(dt=T, t_stop=20)

    source = sim.UserDefinedSource(func_source, name='source')
    beam = sim.TransferFunction(SYS_D.num, SYS_D.den, name='beam')
    switch = sim.UserDefinedFunction(lambda x: 0 if system.t < t_control_start
                                     else x)
    controller = sim.PDC_classic(freq_initial,
                                 lambda x: signal.freqresp(SYS_C, x)[1],
                                 mu)
    combiner = sim.Sum('++')
    recorder = sim.Recorder(name='recorder')

    system.add_blocks(source, beam, controller, combiner, recorder, switch)

    beam.inports[0].connect(combiner.outports[0])
    switch.inports[0].connect(beam.outports[0])
    controller.inports[0].connect(switch.outports[0])
    combiner.inports[0].connect(source.outports[0])
    combiner.inports[1].connect(controller.outports[0])
    recorder.inports[0].connect(beam.outports[0])

    system.log(source.outports[0], 'equivalent disturbance')
    system.log(controller.outports[0], 'controller output')
    system.log(beam.outports[0], 'system output')
    system.run()

    fig = system.logger.plot()
    fig.set_size_inches(8, 6)
    ax = fig.axes[0]
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Magnitude')
    fig.tight_layout()
    ylim = [-3.5, 3.5]  # ax.get_ylim()
    ax.plot([5, 5], ylim, 'r')
    ax.set_ylim(ylim)
    ax.text(5.3, 3.05, 'control starts', color='r')
    ax.legend(loc='lower right')


def test_PDC_improved():
    t_control_start = 5

    system = sim.System(dt=T, t_stop=20)

    source = sim.UserDefinedSource(func_source, name='source')
    beam = sim.TransferFunction(SYS_D.num, SYS_D.den, name='beam')
    switch = sim.UserDefinedFunction(lambda x: 0 if system.t < t_control_start
                                     else x)
    controller = sim.PDC_improved(freq_initial,
                                 lambda x: signal.freqresp(SYS_C, x)[1],
                                 1,1)
    combiner = sim.Sum('++')
    recorder = sim.Recorder(name='recorder')

    system.add_blocks(source, beam, controller, combiner, recorder, switch)

    beam.inports[0].connect(combiner.outports[0])
    switch.inports[0].connect(beam.outports[0])
    controller.inports[0].connect(switch.outports[0])
    combiner.inports[0].connect(source.outports[0])
    combiner.inports[1].connect(controller.outports[0])
    recorder.inports[0].connect(beam.outports[0])

    system.log(source.outports[0], 'equivalent disturbance')
    system.log(controller.outports[0], 'controller output')
    system.log(beam.outports[0], 'system output')
    system.run()

    fig = system.logger.plot()
    fig.set_size_inches(8, 6)
    ax = fig.axes[0]
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Magnitude')
    fig.tight_layout()
    ylim = [-3.5, 3.5]  # ax.get_ylim()
    ax.plot([5, 5], ylim, 'r')
    ax.set_ylim(ylim)
    ax.text(5.3, 3.05, 'control starts', color='r')
    ax.legend(loc='lower right')


show_bode()
test_PDC_classic()
test_PDC_improved()
