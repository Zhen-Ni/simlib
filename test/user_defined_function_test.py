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


def test_CFunction():
    import ctypes as ct
    libname = 'test/test_CFunction.dll'
    types_in = [ct.c_double]
    sizes_in = [NA]
    types_out = [ct.c_double, ct.c_double]
    sizes_out = [NA, NA]
    initfunc = 'initfunc'
    outputstep = 'outputstep'
    blockstep = 'blockstep'
    
    source1 = UserDefinedSource(lambda t: t * 0.5 - 1)
    opt = CFunction(libname, types_in, sizes_in, types_out, sizes_out,
                    initfunc, outputstep, blockstep)
    scope = Recorder(3)
    system = System(dt=0.1, t_stop=1)
    system.add_blocks(source1, opt, scope)
    opt.inports[0].connect(source1.outports[0])
    scope.inports[0].connect(source1.outports[0])
    scope.inports[1].connect(opt.outports[0])
    scope.inports[2].connect(opt.outports[1])
    system.initialize()
    system.run()
    scope.plot()


def test_CFunction2():
    import ctypes as ct
    libname = 'test/test_CFunction.dll'
    types_in = [ct.c_double]
    sizes_in = [2]
    types_out = [ct.c_double]
    sizes_out = [2]
    initfunc = 'initfunc2'
    outputstep = 'outputstep2'
    blockstep = 'blockstep2'
    
    source1 = UserDefinedSource(lambda t: [1,t])
    opt = CFunction(libname, types_in, sizes_in, types_out, sizes_out,
                    initfunc, outputstep, blockstep)
    scope = Recorder(4)
    system = System(dt=0.1, t_stop=1)
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
    test_CFunction()
    test_CFunction2()
