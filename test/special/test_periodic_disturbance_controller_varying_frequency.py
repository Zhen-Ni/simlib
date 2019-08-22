#!/usr/bin/env python3

# This example origins from my paper published at ICSV26

import os
import sys
os.chdir('../..')
sys.path.append(os.curdir)

import numpy as np
import simlib as sim
#from control_algorithm import *

from pylab import *
plt.rc('font', family='Times New Roman',size=12)

import scipy.signal as signal
NUM_D = (6.41954567e+04, 5.15243447e+05, 4.02737252e+10, 1.56801855e+11,
         4.65854675e+15, 6.61300105e+15, 1.12177600e+20)
DEN_D = (1.00000000e+00, 9.00847856e+00, 8.32178625e+05, 4.31468236e+06,
         1.80095144e+11, 5.41027517e+11, 1.28211348e+16, 1.56021332e+16,
         1.76946903e+20)
NUM_C = (1.47339605e+05, 1.30964156e+06, 1.22021254e+11, 3.58259584e+11,
         1.57216451e+16, 8.73410843e+15, 1.98202392e+20)
DEN_C = (1.00000000e+00, 9.23877068e+00, 8.32219424e+05, 4.51335402e+06,
         1.80125293e+11, 5.90103504e+11, 1.28263943e+16, 1.92583915e+16,
         1.77187179e+20)
SYS_C = signal.TransferFunction(NUM_C, DEN_C)
func_response = lambda x,SYS_C=SYS_C: signal.freqresp(SYS_C,x)[1]
SYS_D = signal.TransferFunction(NUM_D, DEN_D)
T = 0.001   # sampling frequency
SYS_C = sim.sample_system(SYS_C, T)
SYS_D = sim.sample_system(SYS_D, T)


# 等效干扰信号
class FuncSource:
    def __init__(self):
        self.theta1 = 25**2/2/pi
        self.theta2 = 50**2/2/pi
        self.theta3 = 75**2/2/pi
        self.freq0 = np.array([25,50,75])
#        self.freqn = np.array([24.9,24.9*2,24.9*3])
        self.freqn = np.array([40,35,90])
        self.freq_history = []

    def __call__(self,t):
        y1 = np.sin(2*np.pi*self.theta1)
        y2 = np.sin(2*np.pi*self.theta2)
        y3 = np.sin(2*np.pi*self.theta3)
        freqs = self.get_freqs(t)
        self.freq_history.append(freqs)

        self.theta1 += freqs[0] * T
        self.theta2 += freqs[1] * T
        self.theta3 += freqs[2] * T

        return y1 + y2 + y3

    def get_freqs(self, t):
        freq0 = self.freq0
        freqn = self.freqn
        t0 = 40
        tn = 60
        if t < t0:
            return freq0
        elif t < tn:
            return freq0+(t-t0)*(freqn-freq0)/(tn-t0)
        else:
            return freqn



def build_system_mine(func_source):
    system = sim.System(dt=T, t_stop=90)

    source = sim.UserDefinedSource(func_source, name='source')
    noise = sim.GaussianNoise(0.04**2, seed=201905081338)
    source_total = sim.Sum('++')
    beamc = sim.TransferFunction(SYS_C.num, SYS_C.den, name='beamc')
    beamd = sim.TransferFunction(SYS_D.num, SYS_D.den, name='beamd')
    beamd_noise = sim.TransferFunction(SYS_D.num, SYS_D.den, name='beamd-noise')
    controller = sim.PDC(3, func_response,0.6, 15, trans_omega='metric',t_fft_start=9,t_fft=1,resolution=0.01)
    combiner = sim.Sum('++')
    recorder = sim.Recorder(name ='recorder')

    system.add_blocks(source, beamc, beamd,controller, combiner,recorder,noise, source_total, beamd_noise)

    beamc.inports[0].connect(controller.outports[0])
    controller.inports[0].connect(combiner.outports[0])
    beamd.inports[0].connect(source_total.outports[0])
    combiner.inports[0].connect(beamd.outports[0])
    combiner.inports[1].connect(beamc.outports[0])
    recorder.inports[0].connect(combiner.outports[0])
    source_total.inports[0].connect(source.outports[0])
    source_total.inports[1].connect(noise.outports[0])
    beamd_noise.inports[0].connect(noise.outports[0])

    system.log(beamd.outports[0],'w/o control')
    system.log(combiner.outports[0],'with control')
    system.log(beamd_noise.outports[0],'filtered noise')
    return system



def plot_my_algorithm():
    func_source = FuncSource()
    system = build_system_mine(func_source)
    ws = []
    system.callback=lambda: ws.append(system.blocks[3].w.copy())
    system.run()
    ws = np.array(ws)

    fig = system.logger.plot()
    fig.set_size_inches(4,3)
    ax = fig.axes[0]
    for line in ax.lines:
        line.set_alpha(0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    fig.tight_layout()
    ylim =  [-4,4]#ax.get_ylim()
    ax.plot([10,10],ylim,'r')
    ax.plot([40,40],ylim, ':m')
    ax.plot([60,60],ylim, ':m')
    ax.set_ylim(ylim)
    ax.text(10.5,3.0,'Control starts', color='r')
    ax.legend(loc='lower right', framealpha=0.8)

    t = system.logger.t.data
    freqs = np.array(func_source.freq_history)
    idx = [2,0,1]
    for i in range(3):
        fig = figure(figsize=(4,3))
        ax = fig.add_subplot(111)
        ax.plot(t[t>=10], ws[:,i][t>=10]/2/pi, label='Estimated',lw=2)
        ax.plot(t[t>=10],freqs[:,idx[i]][t>=10], '--',label='True',lw=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ylim = ax.get_ylim()
        ax.plot([10,10],ylim,'r')
        ax.plot([40,40],ylim, ':m')
        ax.plot([60,60],ylim, ':m')
        ax.set_ylim(ylim)
        ax.legend(loc='lower left', framealpha=0.8)
        ax.set_xlim(t[0],t[-1])
        ax.grid(which='both')
        fig.tight_layout()

    return system



def plot_spectrogram(system, name):
    y = system.logger['with control'].data
    f, t, Sxx = signal.spectrogram(y, 1/T, window='blackman',nperseg=1000,noverlap=900,nfft=16384)
    fig = figure(figsize=(4,3))
    ax = fig.add_subplot(111)
    surf = ax.pcolormesh(*meshgrid(t, f[f<100]), 10*log10(Sxx[f<100]),cmap='jet', vmin=-50, vmax=0)
    fig.colorbar(surf)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    xlim = 0,90
    ylim = 0,100
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.plot([10,10],ylim, 'r')
    ax.plot([40,40],ylim, ':m')
    ax.plot([60,60],ylim, ':m')
    fig.tight_layout()


system = plot_my_algorithm()
plot_spectrogram(system, 'present')
