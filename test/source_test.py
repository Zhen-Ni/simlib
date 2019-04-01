#!/usr/bin/env python3

import os
os.chdir('..')

from simlib import *


def test_Clock():
    source = Clock()
    system = System(dt=0.1)
    system.add_blocks(source)
    system.log(source.Out[0])
    system.initialize()
    system.run()
    system.logger.plot()
    return system


def test_SineWave():
    source = SineWave(frequency=1)
    system = System(dt=0.1)
    system.add_blocks(source)
    system.log(source.Out[0])
    system.initialize()
    system.run()
    system.logger.plot()
    return system


def test_Impulse():
    source = Impulse(n0=0)
    system = System(dt=0.1)
    system.add_blocks(source)
    system.log(source.Out[0])
    system.initialize()
    system.run()
    system.logger.plot()
    return system


def test_Step():
    source = Step()
    system = System(dt=0.1)
    system.add_blocks(source)
    system.log(source.Out[0])
    system.initialize()
    system.run()
    system.logger.plot()
    return system


def test_Constant():
    source = Constant()
    system = System(dt=0.1)
    system.add_blocks(source)
    system.log(source.Out[0])
    system.initialize()
    system.run()
    system.logger.plot()
    return system


def test_RepeatingSequence():
    source = RepeatingSequence()
    system = System(dt=0.1)
    system.add_blocks(source)
    system.log(source.Out[0])
    system.initialize()
    system.run()
    system.logger.plot()
    return system


def test_UserDefinedSource():
    source = UserDefinedSource(lambda t: t**2)
    system = System(dt=0.1)
    system.add_blocks(source)
    system.log(source.Out[0])
    system.initialize()
    system.run()
    system.logger.plot()
    return system


def test_GaussianNoise():
    source = GaussianNoise()
    system = System(dt=0.1)
    system.add_blocks(source)
    system.log(source.Out[0])
    system.initialize()
    system.run()
    system.logger.plot()
    return system


if __name__ == '__main__':
    test_Clock()
    test_SineWave()
    test_Impulse()
    test_Step()
    test_Constant()
    test_RepeatingSequence()
    test_UserDefinedSource()
    test_GaussianNoise()
