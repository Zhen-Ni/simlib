#!/usr/bin/env python3

import os
import sys
os.chdir('..')
sys.path.append(os.curdir)

from simlib import *


def test_UserDefinedFunction():
    source1 = UserDefinedSource(lambda t: t * 0.5 - 1)
    opt = UserDefinedFunction(lambda x, t: (
        2 * x, 2 * t), t_as_arg=True, nout=2)
    scope = Recorder(3)
    system = System(dt=0.1)
    system.add_blocks(source1, opt, scope)
    opt.inports[0].connect(source1.outports[0])
    scope.inports[0].connect(source1.outports[0])
    scope.inports[1].connect(opt.outports[0])
    scope.inports[2].connect(opt.outports[1])
    system.initialize()
    system.run()
    scope.plot()


def test_PythonFunction():
    def initfunc():
        print('initfunc')

    def outputstep(portid):
        if outputstep.printed:
            return NA
        print('portid:', portid)
        outputstep.printed = True
        return NA
    outputstep.printed = False

    source1 = UserDefinedSource(lambda t: t * 0.5 - 1)
    opt = PythonFunction(initfunc=initfunc, outputstep=outputstep,
                         blockstep=lambda x: (2 * x,))
    scope = Recorder(2)
    system = System(dt=0.1)
    system.add_blocks(source1, opt, scope)
    opt.inports[0].connect(source1.outports[0])
    scope.inports[0].connect(source1.outports[0])
    scope.inports[1].connect(opt.outports[0])
    system.initialize()
    system.run()
    scope.plot()


if __name__ == '__main__':
    test_UserDefinedFunction()
    test_PythonFunction()
