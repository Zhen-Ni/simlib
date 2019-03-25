#!/usr/bin/env python3

import os
import sys
os.chdir('..')
sys.path.append(os.curdir)

from simlib import *


def test_Sum():
    source1 = UserDefinedSource(lambda t: t * 0.5 - 1)
    source2 = RepeatingSequence()
    opt = Sum('++')
    scope = Recorder(3)
    system = System(dt=0.1)
    system.add_blocks(source1, source2, opt, scope)
    opt.inports[0].connect(source1.outports[0])
    opt.inports[1].connect(source2.outports[0])
    scope.inports[0].connect(source1.outports[0])
    scope.inports[1].connect(source2.outports[0])
    scope.inports[2].connect(opt.outports[0])
    system.initialize()
    system.run()
    scope.plot()


def test_Gain():
    source1 = UserDefinedSource(lambda t: t * 0.5 - 1)
    opt = Gain(2)
    scope = Recorder(2)
    system = System(dt=0.1)
    system.add_blocks(source1, opt, scope)
    opt.inports[0].connect(source1.outports[0])
    scope.inports[0].connect(source1.outports[0])
    scope.inports[1].connect(opt.outports[0])
    system.initialize()
    system.run()
    scope.plot()


def test_DotProduct():
    source1 = UserDefinedSource(lambda t: t * 0.5 - 1)
    source2 = RepeatingSequence()
    opt = DotProduct(2)
    scope = Recorder(3)
    system = System(dt=0.1)
    system.add_blocks(source1, source2, opt, scope)
    opt.inports[0].connect(source1.outports[0])
    opt.inports[1].connect(source2.outports[0])
    scope.inports[0].connect(source1.outports[0])
    scope.inports[1].connect(source2.outports[0])
    scope.inports[2].connect(opt.outports[0])
    system.initialize()
    system.run()
    scope.plot()




if __name__ == '__main__':
    test_Sum()
    test_Gain()
    test_DotProduct()

