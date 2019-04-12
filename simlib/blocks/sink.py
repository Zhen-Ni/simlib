"""sink.py

This module contains the basic sinks in control system.
"""

import numpy as np
from ..simsys import BaseBlock
from ..misc import as_uint
from ..simexceptions import SimulationError


__all__ = ['Scope', 'XYGraph', 'Recorder']


def calculate_xylim(xmin, xmax, ymin, ymax, expand=0.1):
    """Calculate the xlim and ylim for figure. The percentage of the size of
    the edge is given by expand."""
    range_x = xmax - xmin
    range_y = ymax - ymin
    range_x *= expand / 2
    range_y *= expand / 2
    if range_x == 0.0:
        range_x = 0.002
    if range_y == 0.0:
        range_y = 0.002
    return xmin - range_x, xmax + range_x, ymin - range_y, ymax + range_y


class Scope(BaseBlock):

    """The Scope block.

    Scope is used to plot signals during simulation.

    Parameters
    ----------
    nin: int
        Number of input signals. (default = 1)

    batch_size: int
        Number of inputs to plot together. (default = 10)

    refresh: bool
        Whether to clear history display when updated. (default = False)

    duration: float or None
        The duration of data to show in the figure. If None, the duration will
        be set to be as long as possible to show all the data. (default to
        None)

    autoscalar: bool, optional
        Whether to scalar the plot  automatically. defaults to False.

    dt: float, optional
        Sampling time of this block. The input and output be evaluate once
        for each sampling time.

    name: string, optional
        The name of this block.

    Ports
    -----
    In[i]:
        The input signals

    """

    def __init__(self, nin=1, batch_size=10, refresh=False, duration=None,
                 autoscalar=False, dt=None, name='Scope'):
        super().__init__(nin=nin, nout=0, dt=dt, name=name)
        self._batch_size = as_uint(batch_size)
        self._refresh = refresh
        self._duration = duration
        self._autoscalar = autoscalar
        import matplotlib.pyplot as plt
        self._plt = plt
#        self._plt.ion()    # Not sure whether it is necessary

    def INITFUNC(self):
        self._time = []
        self._data = []
        self._fig = self._plt.figure()
        self._ax = self._fig.add_subplot(111)
        self._ax.set_xlabel('Time')
        self._ax.grid()

    def BLOCKSTEP(self, *xs):
        self._time.append(self.t)
        signals = []
        for s in xs:
            if not np.iterable(s):
                signals.append([s])
            else:
                signals.append(np.asarray(s).reshape(-1))
        res = np.concatenate(signals)
        self._data.append(res)

        n = len(self._data[0])
        if self.n % self._batch_size == self._batch_size - 1:
            colors = [l.get_color() for l in self._ax.lines]
            self._ax.lines = []
            ymin, ymax = None, None
            for i in range(n):
                y = [j[i] for j in self._data]
                # need to make sure data is not empty
                if self._autoscalar and y:
                    ymin_i = np.min(y)
                    ymax_i = np.max(y)
                    if ymin is None or ymin_i < ymin:
                        ymin = ymin_i
                    if ymax is None or ymax_i > ymax:
                        ymax = ymax_i
                if i < len(colors):
                    self._ax.plot(self._time, y, color=colors[i])
                else:
                    self._ax.plot(self._time, y)

            if self._duration is None:
                self._ax.set_xlim(self._time[0], self._time[-1])
            elif self._duration < self._time[-1] - self._time[0]:
                self._ax.set_xlim(self._time[-1] - self._duration,
                                  self._time[-1])
            else:
                self._ax.set_xlim(self._time[0],
                                  self._time[0] + self._duration)

            if self._autoscalar:
                xmin, xmax, ymin, ymax = calculate_xylim(0, 0, ymin, ymax)
                self._ax.set_ylim(ymin, ymax)

            self._plt.pause(1e-10)          # refresh the figure
            if self._refresh:
                self._time = []
                self._data = []
        return ()


class XYGraph(BaseBlock):

    """The XYGraph block.

    XYGraph is used to plot data arrays with x and y data.

    Parameters
    ----------
    nin: int
        Number of input signals. (default = 1)

    xlabel: string, optional
        Xlabel of the plot.

    ylabel: string, optional
        Ylabel of the plot.

    autoscalar: bool, optional
        Whether to scalar the plot  automatically. defaults to False.

    dt: float, optional
        Sampling time of this block. The input and output be evaluate once
        for each sampling time.

    name: string, optional
        The name of this block.

    Ports
    -----
    In[i]:
        The input xydata, should be iterable with two arrays of the same size.
    """

    def __init__(self, nin=1, xlabel='', ylabel='', autoscalar=False,
                 dt=None, name='XYGraph'):
        super().__init__(nin=nin, nout=0, dt=dt, name=name)
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._autoscalar = autoscalar
        import matplotlib.pyplot as plt
        self._plt = plt
#        self._plt.ion()    # Not sure whether it is necessary

    def INITFUNC(self):
        self._fig = self._plt.figure()
        self._ax = self._fig.add_subplot(111)
        self._ax.set_xlabel(self._xlabel)
        self._ax.set_ylabel(self._ylabel)
        self._ax.grid()

    def BLOCKSTEP(self, *xs):
        colors = [l.get_color() for l in self._ax.lines]
        self._ax.lines = []
        xmin, xmax = None, None
        ymin, ymax = None, None
        has_data = False
        for i, data in enumerate(xs):
            try:
                x, y = data
            except (TypeError, ValueError):
                raise SimulationError('Input to {s} should be iterable with '
                                      'two arrays'.format(s=self))
            # in case the data is empty
            if self._autoscalar and x.size:
                has_data = True
                xmin_i = np.min(x)
                xmax_i = np.max(x)
                ymin_i = np.min(y)
                ymax_i = np.max(y)
                if xmin is None or xmin_i < xmin:
                    xmin = xmin_i
                if xmax is None or xmax_i > xmax:
                    xmax = xmax_i
                if ymin is None or ymin_i < ymin:
                    ymin = ymin_i
                if ymax is None or ymax_i > ymax:
                    ymax = ymax_i
            if i < len(colors):
                self._ax.plot(x, y, color=colors[i])
            else:
                self._ax.plot(x, y)

        if self._autoscalar and has_data:
            xmin, xmax, ymin, ymax = calculate_xylim(xmin, xmax, ymin, ymax)
            self._ax.set_xlim(xmin, xmax)
            self._ax.set_ylim(ymin, ymax)

        self._plt.pause(1e-10)          # refresh the figure
        return ()


class Recorder(BaseBlock):

    """The Recorder block.

    Recorder is used to record and plot signals during simulation.

    Parameters
    ----------
    nin: int, optional
        Number of input signals. (default = 1)

    dt: float, optional
        Sampling time of this block. The input and output be evaluate once
        for each sampling time.

    name: string, optional
        The name of this block.

    Ports
    -----
    In[i]:
        The input signals
    """

    def __init__(self, nin=1, dt=None, name='Recorder'):
        super().__init__(nin=nin, nout=0, dt=dt, name=name)

    def INITFUNC(self):
        self._time = []
        self._data = []

    def BLOCKSTEP(self, *xs):
        self._time.append(self.t)
        signals = []
        for s in xs:
            if not np.iterable(s):
                signals.append([s])
            else:
                signals.append(np.asarray(s).reshape(-1))
        res = np.concatenate(signals)
        self._data.append(res)
        return []

    @property
    def time(self):
        return np.array(self._time)

    @property
    def data(self):
        return np.array(self._data)

    def fetch(self):
        return self.time, self.data

    def plot(self, backend='matplotlib', **kwargs):
        """Plot the cruve shown on the Recorder.

        Parameters
        ----------
        backend: string, optional
            backend used to plot. Either 'matplotlib' or 'matlab'. When
            'backend' is 'matlab', keyword argument 'eng' should be defined.

        **kwargs: keyword arguments, necessary when 'backend' is 'matlab'
            if backend=='matlab', keyword argument 'eng' should be defined
            as an instance of matlab.engine.matlabengine.MatlabEngine class.
        """
        if backend == 'matplotlib':
            import matplotlib.pyplot as plt
            plt.figure()
            try:
                n = len(self._data[0])
            except IndexError:
                if not self.is_initialized():
                    # plot an empty figure
                    n = 0
                else:
                    raise
            for i in range(n):
                plt.step(self._time, [j[i] for j in self._data], where='post')
            plt.xlabel('Time')
            plt.grid()
        elif backend == 'matlab':
            import matlab
            time = matlab.double(self._time)
            data = matlab.double(np.array(self._data).tolist())
            try:
                eng = kwargs['eng']
            except KeyError:
                raise TypeError("required argument 'eng' not found")
            if not isinstance(eng, matlab.engine.matlabengine.MatlabEngine):
                raise TypeError("eng must be MatlabEngine object")
            eng.stairs(time, data)
            eng.xlabel('Time')
            eng.grid(nargout=0)
