#!/usr/bin/env python3


import os
import sys
from pylab import *

os.chdir('..')
sys.path.append(os.curdir)

from simlib import *

def test_FourierTransformer():
    source = UserDefinedSource(lambda t: sin(2*pi*5*t**2))
    sa = FourierTransformer(0, normalize=True)
    bundle = Bundle(2)
    abs_ = UserDefinedFunction(lambda x:np.abs(x))
    graph = XYGraph(1,'freq', dt=0.1)
    scope = Scope(batch_size=10)
    system = System(dt=0.01, t_stop=10)
    system.add_blocks(source, sa, graph, bundle, abs_, scope)
    sa.inports[0].connect(source.outports[0])
    bundle.inports[0].connect(sa.outports[0])
    abs_.inports[0].connect(sa.outports[1])
    bundle.inports[1].connect(abs_.outports[0])
    graph.inports[0].connect(bundle.outports[0])
    scope.inports[0].connect(source.outports[0])
    system.initialize()
    system.run()
    return system

def test_PowerSpectrum():
    source = UserDefinedSource(lambda t: t-5+sin(2*pi*5*t**2))
    sa = PowerSpectrum(100, detrend='linear', scaling='spectrum')
    bundle = Bundle(2)
    graph = XYGraph(1,'freq', dt=0.1)
    scope = Scope(batch_size=10, refresh=False)
    system = System(dt=0.01, t_stop=10)
    system.add_blocks(source, sa, graph, bundle, scope)
    sa.inports[0].connect(source.outports[0])
    bundle.inports[0].connect(sa.outports[0])
    bundle.inports[1].connect(sa.outports[1])
    graph.inports[0].connect(bundle.outports[0])
    scope.inports[0].connect(source.outports[0])
    system.log(sa.outports[1])
    system.initialize()
    system.run()
    return system


if __name__ == '__main__':
#    test_FourierTransformer()
    test_PowerSpectrum()