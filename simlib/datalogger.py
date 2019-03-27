#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt

from .misc import makeplain


class LoggedData():

    """Struct for logged data."""

    def __init__(self, name, data):
        self._name = name
        self._data = data

    def __repr__(self):
        res = 'LoggedData of: {name}'.format(name=self.name)
        return res

    @property
    def name(self):
        return self._name

    @property
    def data(self):
        return self._data


class DataLogger:

    def __init__(self, system: 'System object'):
        self._t = []
        self._ports = []
        self._data = []
        self._names = []
        self._system = system

    @property
    def system(self):
        return self._system

    def __repr__(self):
        res = "DataLogger of {name}".format(name=self._system)
        return res

    def append_signal(self, port: "Port object", name=None):
        if self._t:
            raise ValueError('cannot add port after logged')
        if name is None:
            name = str(port)
        if name == 't':
            raise ValueError('name "t" cannot be used')
        self._ports.append(port)
        self._names.append(name)

    def log(self):
        t = self._system.t
        if len(self._t) and t <= self._t[-1]:
            raise ValueError("t={t} is not larger than the previous one"
                             .format(t=t))
        self._t.append(t)
        self._data.append([])
        for p in self._ports:
            self._data[-1].append(p.get()[0])

    def __getitem__(self, value):
        if value == 't':
            return self.t
        if isinstance(value, str):
            # Use super().__getattribute__ because this function may be called
            # by self.__getattr__. And when using copy.deepcopy, directly using
            # self._names may lead to Errors. Visit
            # "https://stackoverflow.com/questions/40583131/" for more details.
            for i, name in enumerate(super().__getattribute__('_names')):
                if name == value:
                    idx = i
                    break
            else:
                raise KeyError('{name}'.format(name=value))
        else:
            idx = value
        res = [j[idx] for j in self._data]
        return LoggedData(self._names[idx], np.array(res))

    def __getattr__(self, name):
        try:
            return self[name]
        except (KeyError, IndexError):
            raise AttributeError('LoggedData has no value named {name}'
                                 .format(name=name))

    @property
    def ports(self):
        return [p for p in self._ports]

    @property
    def names(self):
        return ['t'] + [n for n in self._names]

    @property
    def t(self):
        return LoggedData('t', np.array(self._t))

    def asplain(self):
        """Extract one-dimensional array from logged data."""
        names = ['t']
        datas = [self._t]
        for i, d in enumerate(self._ports):
            ns, ds = makeplain(self._names[i], [j[i] for j in self._data])
            names += ns
            datas += ds
        return names, datas

    def asdict(self):
        """Extract dictionary from logged data."""
        d = {'t': np.array(self._t)}
        for i in range(len(self._ports)):
            d[self._names[i]] = np.array([line[i] for line in self._data])
        return d

    def savecsv(self, filename):
        names, datas = self.asplain()
        import csv
        if filename.split('.')[-1] != 'csv':
            filename = filename + '.csv'
        with open(filename, 'w', newline='') as csvfile:
            writter = csv.writer(csvfile)
            writter.writerow(names)
            for i in range(len(self._t)):
                writter.writerow(line[i] for line in datas)

    def savemat(self, filename):
        from scipy.io import savemat
        if filename.split('.')[-1] != 'mat':
            filename = filename + '.mat'
        savemat(filename, self.asdict())

    def plot(self, signal_names=None):
        """Use matplotlib to plot logged signals.
        
        By default, this function plots all the logged data if signal_names is
        not given. Users may also specify which signals they want to plot by
        passing the names of these signals in an iterable object by 
        `signal_names`, and the specified signals will be plotted in sequence.
        """
        if signal_names is None:
            names, datas = self.asplain()
        else:
            d = self.asdict()
            names = ['t']
            datas = [d['t']]
            for name in signal_names:
                data = d.get(name)
                if data is None:
                    self.system.warn('cannot find signal named {n}'
                                     .format(n=name))
                    continue
                names.append(name)
                datas.append(data)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(1, len(names)):
            ax.step(datas[0], datas[i], label=names[i], where='post')
        ax.set_xlabel('Time')
        ax.legend()
        ax.grid()

    def simplot(self, name=None, eng=None):
        import matlab
        import sys
#        import os
        if eng is None:
            import matlab.engine
            eng = matlab.engine.start_matlab()
        if name is None:
            name = self.system.name
        args = [name, self._t]
        names, datas = self.asplain()
        for i in range(1, len(names)):
            args.append(str(names[i]))   # use str in case names[i] is None
            args.append(matlab.double(datas[i]))
#        eng.addpath(*sys.path)
#        eng.addpath(os.getcwd() + '\simlib')
        eng.addpath(sys.modules['simlib'].__path__[0])
        eng.simulinkplot(*args, nargout=0)
        return eng
