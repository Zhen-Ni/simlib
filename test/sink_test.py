#!/usr/bin/env python3

import os
import sys
os.chdir('..')
sys.path.append(os.curdir)

from pylab import *
from simlib import *


def test_Scope():
    source1 = UserDefinedSource(lambda t: t * 1.0 - 50)
    source2 = RepeatingSequence()
    scope = Scope(2, dt=0.1, batch_size=100, refresh=True, autoscalar=True)
    system = System(dt=0.1, t_stop=100)
    system.add_blocks(source1, source2, scope)
    scope.inports[0].connect(source1.outports[0])
    scope.inports[1].connect(source2.outports[0])
    system.initialize()
    system.run()


def test_XYGraph():
    source = UserDefinedSource(lambda t: sin(2 * pi * 5 * t**2))
    sa = FourierTransformer(100, normalize=True)
    bundle = Bundle(2)
    abs_ = UserDefinedFunction(lambda x: np.abs(x))
    graph = XYGraph(1, 'freq', autoscalar=False, dt=0.1)
    system = System(dt=0.01, t_stop=10)
    system.add_blocks(source, sa, graph, bundle, abs_)
    sa.inports[0].connect(source.outports[0])
    bundle.inports[0].connect(sa.outports[0])
    abs_.inports[0].connect(sa.outports[1])
    bundle.inports[1].connect(abs_.outports[0])
    graph.inports[0].connect(bundle.outports[0])
    system.initialize()
    system.run()
    return system


def test_Recorder():
    source1 = UserDefinedSource(lambda t: t * 0.5 - 1)
    source2 = RepeatingSequence()
    recorder = Recorder(2, dt=0.1)
    system = System(dt=0.1)
    system.add_blocks(source1, source2, recorder)
    recorder.inports[0].connect(source1.outports[0])
    recorder.inports[1].connect(source2.outports[0])
    system.initialize()
    system.run()
    recorder.plot()


if __name__ == '__main__':
    #    test_Scope()
    test_XYGraph()
#    test_Recorder()
