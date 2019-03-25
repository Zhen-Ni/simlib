#!/usr/bin/env python3

import os
import sys
os.chdir('..')
sys.path.append(os.curdir)

from simlib import *

num = [1, 2, 3]
den = [10, 1, 8]
#den = [1, 0, 0]
eng = None


def test_FIRLMS():
    mu, lenfir = 0.01, 20
    system = System(t_stop=100)
    source = GaussianNoise(seed=0)
    viber = TransferFunction(num, den)
    recorder = Recorder(2)
    lms = FIRLMS(length=lenfir, mu=mu)
    comp = Sum('+-')
    system.add_blocks(source, viber, recorder, lms, comp)
    # build up the system
    viber.inports[0].connect(source.outports[0])
    recorder.inports[0].connect(viber.outports[0])
    lms.inports[0].connect(source.outports[0])
    comp.inports[0].connect(viber.outports[0])
    comp.inports[1].connect(lms.outports[0])
    lms.inports[1].connect(comp.outports[0])
    recorder.inports[1].connect(lms.outports[0])
    # log data
    system.log(viber.outports[0], name='system output')
    system.log(lms.outports[0], name='lms output')
    system.log(comp.outports[0], name='error')
    # run
    system.run()
#    recorder.plot()
    system.logger.plot()
    # matlab plot
#    return system.logger.simplot(name='FIRLMS', eng=eng)


def test_FIRNLMS():
    mu, lenfir = 0.01, 20
    system = System(t_stop=10)
    source = GaussianNoise(seed=0)
    viber = TransferFunction(num, den)
    recorder = Recorder(2)
    lms = FIRNLMS(length=lenfir)
    comp = Sum('+-')
    system.add_blocks(source, viber, recorder, lms, comp)
    # build up the system
    viber.inports[0].connect(source.outports[0])
    recorder.inports[0].connect(viber.outports[0])
    lms.inports[0].connect(source.outports[0])
    comp.inports[0].connect(viber.outports[0])
    comp.inports[1].connect(lms.outports[0])
    lms.inports[1].connect(comp.outports[0])
    recorder.inports[1].connect(lms.outports[0])
    # log data
    system.log(viber.outports[0], name='system output')
    system.log(lms.outports[0], name='lms output')
    system.log(comp.outports[0], name='error')
    # run
    system.run()
    recorder.plot()
    return system


def test_FIRAPLMS():
    mu, lenfir = 0.01, 20
    system = System(t_stop=10)
    source = GaussianNoise(seed=0)
    viber = TransferFunction(num, den)
    recorder = Recorder(2)
    lms = FIRAPLMS(length=lenfir, m=3)
    comp = Sum('+-')
    system.add_blocks(source, viber, recorder, lms, comp)
    # build up the system
    viber.inports[0].connect(source.outports[0])
    recorder.inports[0].connect(viber.outports[0])
    lms.inports[0].connect(source.outports[0])
    comp.inports[0].connect(viber.outports[0])
    comp.inports[1].connect(lms.outports[0])
    lms.inports[1].connect(comp.outports[0])
    recorder.inports[1].connect(lms.outports[0])
    # log data
    system.log(viber.outports[0], name='system output')
    system.log(lms.outports[0], name='lms output')
    system.log(comp.outports[0], name='error')
    # run
    system.run()
    recorder.plot()
    # matlab plot
    return system.logger.simplot(name='APLMS', eng=eng)


def test_FIRTDLMS():
    mu, lenfir = 0.01, 20
    system = System(t_stop=10)
    source = GaussianNoise(seed=0)
    viber = TransferFunction(num, den)
    recorder = Recorder(2)
    lms = FIRTDLMS(length=lenfir, mu=mu, beta=0.98)
    comp = Sum('+-')
    system.add_blocks(source, viber, recorder, lms, comp)
    # build up the system
    viber.inports[0].connect(source.outports[0])
    recorder.inports[0].connect(viber.outports[0])
    lms.inports[0].connect(source.outports[0])
    comp.inports[0].connect(viber.outports[0])
    comp.inports[1].connect(lms.outports[0])
    lms.inports[1].connect(comp.outports[0])
    recorder.inports[1].connect(lms.outports[0])
    # log data
    system.log(viber.outports[0], name='system output')
    system.log(lms.outports[0], name='lms output')
    system.log(comp.outports[0], name='error')
    # run
    system.run()
    recorder.plot()
    # matlab plot
    return system.logger.simplot(name='FIRTDLMS', eng=eng)


def test_IIRLMS():
    mu, N, M = 0.01, 3, 3
    system = System(t_stop=100)
    source = GaussianNoise(seed=0)
    viber = TransferFunction(num, den)
    recorder = Recorder(2)
    comp = Sum('+-')
    system.add_blocks(source, viber, recorder, lms, comp)
    # build up the system
    viber.inports[0].connect(source.outports[0])
    recorder.inports[0].connect(viber.outports[0])
    lms.inports[0].connect(source.outports[0])
    comp.inports[0].connect(viber.outports[0])
    comp.inports[1].connect(lms.outports[0])
    lms.inports[1].connect(comp.outports[0])
    recorder.inports[1].connect(lms.outports[0])
    # log data
    system.log(viber.outports[0], name='system output')
    system.log(lms.outports[0], name='lms output')
    system.log(comp.outports[0], name='error')
    # run
    system.run()
    recorder.plot()
    # matlab plot
#    return system.logger.simplot(name='IIRLMS', eng=eng)


def test_FIRRLS():
    lambda_, lenfir = 0.9, 20
    system = System(t_stop=10)
    source = GaussianNoise(seed=0)
    viber = TransferFunction(num, den)
    recorder = Recorder(2)
    rls = FIRRLS(length=lenfir, lambda_=lambda_)
    comp = Sum('+-')
    system.add_blocks(source, viber, recorder, rls, comp)
    # build up the system
    viber.inports[0].connect(source.outports[0])
    recorder.inports[0].connect(viber.outports[0])
    rls.inports[0].connect(source.outports[0])
    comp.inports[0].connect(viber.outports[0])
    comp.inports[1].connect(rls.outports[0])
    rls.inports[1].connect(comp.outports[0])
    recorder.inports[1].connect(rls.outports[0])
    # log data
    system.log(viber.outports[0], name='system output')
    system.log(rls.outports[0], name='rls output')
    system.log(comp.outports[0], name='error')
    # run
    system.run()
#    recorder.plot()
    system.logger.plot()
    # matlab plot
    return system.logger.simplot(name='FIRLMS', eng=eng)


if __name__ == '__main__':
    #    import matlab.engine
    #    eng = matlab.engine.start_matlab()
    res = test_FIRLMS()
#    sys1 = test_FIRNLMS()
#    sys2 = test_FIRAPLMS()
#    res = test_FIRTDLMS()
#    res = test_IIRLMS()
#    res = test_FIRRLS()
