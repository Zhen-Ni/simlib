import os
import sys
os.chdir('..')
sys.path.append(os.curdir)

from simlib import *


def test_Mux():
    source1 = UserDefinedSource(lambda t: [t, 0.5 * t])
    source2 = Step()
    mux = Mux(2,)
    recorder = Recorder()
    system = System()
    system.add_blocks(source1, source2, mux, recorder)
    mux.inports[0].connect(source1.outports[0])
    mux.inports[1].connect(source2.outports[0])
    recorder.inports[0].connect(mux.outports[0])
    system.initialize()
    system.run()
    recorder.plot()


def test_Demux():
    source = UserDefinedSource(lambda t: [2 * t, 1 * t, 0.5 * t])
    demux = Demux(4, default=1)
    recorder = Recorder(4)
    system = System()
    system.add_blocks(source, demux, recorder)
    demux.inports[0].connect(source.outports[0])
    recorder.inports[0].connect(demux.outports[0])
    recorder.inports[1].connect(demux.outports[1])
    recorder.inports[2].connect(demux.outports[2])
    recorder.inports[3].connect(demux.outports[3])
    system.initialize()
    system.run()
    recorder.plot()


if __name__ == '__main__':
    test_Mux()
    test_Demux()
