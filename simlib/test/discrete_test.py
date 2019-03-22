#!/usr/bin/env python3

import os
import sys
os.chdir('..')
sys.path.append(os.curdir)

from simlib import *


def test_Delay():
    source = UserDefinedSource(lambda t: t)
    component = Delay(2, initial=[5, 1])
    recorder = Recorder(2)
    system = System(dt=1)
    system.add_blocks(source, component, recorder)
    component.inports[0].connect(source.outports[0])
    recorder.inports[0].connect(source.outports[0])
    recorder.inports[1].connect(component.outports[0])
    system.initialize()
    system.run()
    recorder.plot()


def test_TappedDelay():
    source = UserDefinedSource(lambda t: t)
    component = TappedDelay(2, initial=[5,1])
    recorder = Recorder(2)
    system = System(dt=1)
    system.add_blocks(source, component, recorder)
    component.inports[0].connect(source.outports[0])
    recorder.inports[0].connect(source.outports[0])
    recorder.inports[1].connect(component.outports[0])
    system.initialize()
    system.run()
    recorder.plot()


def test_FIRFilter():
    source = UserDefinedSource(lambda t: t)
    component = FIRFilter([2,0,0,1], initial_inputs=[3, 2, 1])
    recorder1 = Recorder(2)
    system = System(dt=1, t_stop=5)
    system.add_blocks(source, component, recorder1)
    component.inports[0].connect(source.outports[0])
    recorder1.inports[0].connect(source.outports[0])
    recorder1.inports[1].connect(component.outports[0])
    system.initialize()
    system.run()
    recorder1.plot()


def test_FIRFilterTimeVarying():
    source = UserDefinedSource(lambda t: t)
    coefficients = Constant([2,0,0,1])
    component = FIRFilterTimeVarying(4, initial_inputs=[3, 2, 1])
    recorder1 = Recorder(2)
    system = System(dt=1, t_stop=5)
    system.add_blocks(source, component,coefficients, recorder1)
    component.inports[0].connect(source.outports[0])
    component.inports[1].connect(coefficients.outports[0])
    recorder1.inports[0].connect(source.outports[0])
    recorder1.inports[1].connect(component.outports[0])
    system.initialize()
    system.run()
    recorder1.plot()

def test_IIRFilter():
    source = UserDefinedSource(lambda t: t)
    component = IIRFilter([0,1,5,7,8],[2,1,3,6])
    recorder1 = Recorder(2)
    system = System(dt=1, t_stop=5)
    system.add_blocks(source, component, recorder1)
    component.inports[0].connect(source.outports[0])
    recorder1.inports[0].connect(source.outports[0])
    recorder1.inports[1].connect(component.outports[0])
    system.initialize()
    system.run()
    recorder1.plot()

def test_IIRFilterTimeVarying():
    source = UserDefinedSource(lambda t: t)
    num = Constant([0,1,5,7,8])
    den = Constant([2,1,3,6])
    component = IIRFilterTimeVarying(5,4)
    recorder1 = Recorder(2)
    system = System(dt=1, t_stop=5)
    system.add_blocks(source, component, recorder1, num, den)
    component.inports[0].connect(source.outports[0])
    component.inports[1].connect(num.outports[0])
    component.inports[2].connect(den.outports[0])
    recorder1.inports[0].connect(source.outports[0])
    recorder1.inports[1].connect(component.outports[0])
    system.initialize()
    system.run()
    recorder1.plot()


def test_TransferFunction():
    source = UserDefinedSource(lambda t: t)
    component = TransferFunction(
        [0, 0, 0, 1, 0], [0,0,1,2,1])
    recorder1 = Recorder(2)
    system = System(dt=1, t_stop=5)
    system.add_blocks(source, component, recorder1)
    component.inports[0].connect(source.outports[0])
    recorder1.inports[0].connect(source.outports[0])
    recorder1.inports[1].connect(component.outports[0])
    system.initialize()
    system.run()
    recorder1.plot()


def test_StateSpace():
    source = UserDefinedSource(lambda t: t)
    G = [[0, 1], [0, 0]]
    H = [0, 1]
    C = [[1, 0], [0, 1]]
    D = 0, 0
    component = StateSpace(G, H, C, D, initial=1)
    recorder1 = Recorder(2)
    recorder2 = Recorder()
    system = System(dt=1)
    system.add_blocks(source, component, recorder1, recorder2)
    component.inports[0].connect(source.outports[0])
    recorder1.inports[0].connect(source.outports[0])
    recorder1.inports[1].connect(component.outports[0])
    recorder2.inports[0].connect(component.outports[1])
    system.initialize()
    system.run()
    recorder1.plot()
    recorder2.plot()


if __name__ == '__main__':
    test_Delay()
    test_TappedDelay()
    test_FIRFilter()
    test_FIRFilterTimeVarying()
    test_IIRFilter()
    test_IIRFilterTimeVarying()
    test_TransferFunction()
    test_StateSpace()
