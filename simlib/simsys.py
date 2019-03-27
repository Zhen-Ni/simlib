"""simsys.py

The simsys module contains the basic structures to build a simulation system.
"""

import copy
import sys
from .datalogger import DataLogger
from .simexceptions import SetupError, InitializationError, StopSimulation, \
    StepError, DefinitionError

__all__ = ['System', 'BaseBlock', 'NA']

STEPRATETOL = 1e-6


class _NAType(float):
    def __repr__(self):
        return 'NA'

    def __str__(self):
        return 'NA'


NA = _NAType('nan')


class System:

    """The Simulation system.

    A Simulation system must be built for each control structure.

    Parameters
    ----------
    t_start: float, optional
        Start time of the system.

    t_stop: float, optional
        Stop time of the system. Simulation will continue only when t < t_stop.

    dt: float, optional
        Sampling time of this block. The input and output be evaluate once
        for each sampling time.

    callback: callable, optional
        Callback function. Will be called after each iteration.        

    name: string, optional
        The name of this system.
    """

    def __init__(self, t_start=0, t_stop=10, dt=0.01, callback=None,
                 name='System'):
        """Initlialize the simulation system."""
        self._t_start = t_start
        self._t_stop = t_stop
        self._dt = dt
        self._n = None    # current number of step
        self._blocks = []
        self._initialized = False
        self._name = name
        self._logger = None
        self._callback_function = callback
        self._warnings = []

    def is_initialized(self):
        """Check whether the system has been initialized."""
        return self._initialized

    @property
    def n(self):
        return self._n

    @property
    def t(self):
        try:
            return self._n * self._dt + self._t_start    # simulation time
        except TypeError:
            # if unitialized
            if self._n is None:
                return None
            else:
                raise

    @property
    def blocks(self):
        return tuple(self._blocks)

    @property
    def name(self):
        return self._name

    @property
    def system(self):
        return self

    @property
    def t_start(self):
        return self._t_start

    @t_start.setter
    def t_start(self, value):
        if self._initialized:
            raise SetupError('cannot set start time after initialization')
        self._t_start = value

    @property
    def t_stop(self):
        return self._t_stop

    @t_stop.setter
    def t_stop(self, value):
        if self._initialized:
            raise SetupError('cannot set stop time after initialization')
        self._t_stop = value

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        if self._initialized:
            raise SetupError('cannot set sampling time after initialization')
        self._dt = value

    @property
    def logger(self):
        return self._logger

    @property
    def callback(self):
        return self._callback_function

    @callback.setter
    def callback(self, callback=None):
        self._callback_function = callback

    @property
    def warnings(self):
        return self._warnings

    def add_block(self, block):
        """Add a block to the simulation system."""
        if block in self.blocks:
            raise SetupError('{block} has already in {system}'.format(
                             block=block, system=self))
        self._blocks.append(block)
        block._system = self
        return block

    def add_blocks(self, *blocks):
        """Add blocks to the control system one by one."""
        for b in blocks:
            self.add_block(b)
        return blocks

    def _assert_initialization(self):
        try:
            assert(self._initialized)
        except AssertionError:
            raise InitializationError('{name} has not been initialized'
                                      .format(name=self))

    def initialize(self):
        if not self._initialized:
            self._do_initialize()
        else:
            self.warn('{s} has been initialized'.format(s=self))

    def _do_initialize(self):
        """Initilize the system."""
        # generate a logger if it not exists
        if self._logger is None:
            self._logger = DataLogger(self)
        self._n = -1
        for b in self._blocks:
            b.initialize()
        self._initialized = True
        self.step_forward()  # Step to n = 0

    def reset(self):
        """Reset the system to initial state."""
        self._n = None
        self._logger = None
        for b in self._blocks:
            b.reset()
        self._initialized = False

    def __repr__(self):
        res = '{name}'.format(name=self.name)
        return res

    def step_forward(self):
        """Simulate one step.

        The procedure is as follows:
        (1) check whether the system has been initialized
        (2) check whether the simulation time reaches the end
        (3) iter all the blocks
        (4) log data for this time step
        (5) call callback function if it is available
        """
        self._assert_initialization()
        if self.t < self.t_stop:
            self._n += 1
            for block in self._blocks:
                block.step()
            self._log_step()
            if self._callback_function:
                self._callback_function()
        else:
            raise StopSimulation

    def run(self):
        """Start system simulation."""
        if not self._initialized:
            self.initialize()
        try:
            while True:
                self.step_forward()
        except StopSimulation:
            pass

    def log(self, signal, name=None):
        """Choose a signal to log during simulation."""
        if self._initialized:
            raise SetupError('cannot set log data after initialization')
        if self._logger is None:
            self._logger = DataLogger(self)
        self._logger.append_signal(signal, name)

    def _log_step(self):
        self._logger.log()

    def warn(self, msg, warn_once=False):
        """Handling warning message generated by the (blocks of the) system.

        Parameters
        ----------
        msg: str
            The warning message.

        warn_once: bool, optional
            Whether to ignore the warning message if it shows up more than
            once. (default to False)
        """
        if warn_once and (msg in self._warnings):
            return
        sys.stderr.write('Warning: ' + msg + '\n')
        self._warnings.append(msg)


class Port:

    """Signal transfer component for Blocks."""

    def __init__(self, parent=None):
        self._parent = parent
        self.name = None
        self._value = None
        self._n = -1

    @property
    def parent(self):
        return self._parent

    @property
    def system(self):
        return self._parent.system

    def rename(self, name):
        self.name = name

    def set(self, value, n):
        self._value = copy.copy(value)
        if self._n != n:
            self._n = n
        else:
            pass

    def get(self):
        return self._value, self._n

    def reset(self):
        self._n = -1
        self._value = None

    def __str__(self):
        if self.name is not None:
            res = "{parent}.{name}".format(parent=self._parent, name=self.name)
        else:
            res = repr(self)
        return res

    def __repr__(self):
        res = '{parent}.Port'.format(parent=self._parent)
        if self.name is not None:
            res += ': ' + self.name
        return res


class InPort(Port):

    """Input port of a block.
        The values are set by self when asked for update, got by parent"""

    def __init__(self, parent=None, id=None):
        super().__init__(parent)
        self._id = id
        self._connector = None  # the outport connected to
        self.name = None

    def set(self, value, n):
        self._value = value
        if self._n != n:
            self._n = n
        else:
            pass

    @property
    def id(self):
        return self._id

    def connect(self, outport):
        if self.system is None or outport.system is None:
            raise SetupError('at least one of the blocks between {self} and '
                             '{other} is not connect to a system'
                             .format(self=self.parent, other=outport.parent))
        if self._parent.is_initialized():
            raise SetupError('the block has been initialized and cannot '
                             'be changed')
        if self.connector:
            self.system.warn('"{self}" has been connected to "{connector}",'
                             ' connect to "{outport}" instead'
                             .format(self=self, connector=self.connector,
                                     outport=outport))
        if outport.system != self.system:

            self.system.warn('"{connector}" and "{self}" do not belong to '
                             'the same system'
                             .format(self=self, connector=self.connector,
                                     outport=outport))
        self._connector = outport

    @property
    def connector(self):
        return self._connector

    def _update(self):
        """Update data from connector."""
        self.set(*self._connector.get())

    def __repr__(self):
        res = "{parent}.In[{id}]".format(parent=self._parent, id=self._id)
        if self.name is not None:
            res += ': ' + self.name
        return res

    def step(self):
        """Get newest data from connector."""
        self._step()
        return self.get()[0]

    def _step(self):
        """Update newest data from connector.
        current step number n = self.system.n
        Raises StepError if fail to get the value of step n."""
        n = self.system.n
        if n == self._n:
            return

        try:
            self._connector._step()

        # when this port is not connected with others
        except AttributeError:
            if self._connector is None:
                self.set(NA, n)
                return
            else:
                raise

        self._update()

        # This should never happen
        if self._n != n:
            raise StepError('step {n} of {name} failed.'
                            .format(n=n, name=self))


class OutPort(Port):

    """Output port of a block.
    The values are set by parent, got by inPort"""

    def __init__(self, parent=None, id=None):
        super().__init__(parent)
        self._id = id
        self.name = None

    def __repr__(self):
        res = "{parent}.Out[{id}]".format(parent=self._parent, id=self._id)
        if self.name is not None:
            res += ': ' + self.name
        return res

    def set(self, value, n):
        if value is None:
            self.system.warn("port {p} gets None as input, do you mean sim.NA?"
                             .format(p=self), warn_once=True)
        self._value = copy.copy(value)
        if self._n != n:
            self._n = n
        else:
            self.system.warn("port {p} at step {n} has "
                             "already been set.".format(p=self, n=n))

    @property
    def id(self):
        return self._id

    def connect(self, inPort):
        inPort.connect(self)

    def _step(self):
        """Get latest data from parent.

        Should only be called from InPort instance."""
        # if needs step, parent will set all the values of this port
        # After this operation, self.n should be equal to self.system.n
        n = self.system._n            # system must be initialized first
        if self._n < n:
            # try to ask parent to update this output without a complete step
            self._parent.output_step(self.id)
            if self._n < n:
                # parent needs a complete step for this output
                self._parent.step()

        # This should never happen!
        elif self._n > n:
            raise StepError('{name} has stepped over {n}, '
                            'with n={sn}'.format(name=self, n=n, sn=self._n))


class BaseBlock:

    """The base class of block modules.

    A block is a component in the control circuit. The input and output of a
    block is updated every sampling time. Each input or output can be a number
    or an array.

    Parameters
    ----------
    nin: int, optional
        Number of inputs of this block.

    nout: int, optional
        Number of outputs of this block.

    dt: float, optional
        Sampling time of this block. The input and output be evaluate once
        for each sampling time.

    name: string, optional
        The name of this block.
    """

    def __init__(self, nin=1, nout=1, dt=None, name='Base Block'):
        self._initialized = False

        # number of system steps
        # should be the same with self.system.n after each system step
        self._n = None
        # number of real steps, dependent on the block's sampling time
        # should be self._n if self.dt = self.system.dt
        self._n_block = None

        self._name = name
        self._parent = None
        self._system = None
        self.set_dt(dt)      # set sampling time
        self._stepRate = None  # relative sampling frequency compared to system
        self._inports = tuple([InPort(parent=self, id=i)
                               for i in range(nin)])
        self._outports = tuple([OutPort(parent=self, id=i)
                                for i in range(nout)])

    def __repr__(self):
        res = '{name}'.format(name=self._name)
        return res

    @property
    def system(self):
        return self._system

    @property
    def dt(self):
        if self._dt is None or self._dt <= 0:
            return self.system.dt
        else:
            return self._dt

    @dt.setter
    def dt(self, value):
        return self.set_dt(value)

    @property
    def t_start(self):
        return self.system.t_start

    @property
    def t_stop(self):
        return self.system.t_stop

    @property
    def n(self):
        return self._n

    @property
    def t(self):
        """Time of this block.

        May be different from the self.system.t if sampling time
        is different."""
        return self._n_block * self.dt + self.t_start

    @property
    def In(self):
        return self._inports

    @property
    def Out(self):
        return self._outports

    @property
    def inports(self):
        return self._inports

    @property
    def outports(self):
        return self._outports

    def is_initialized(self):
        return self._initialized

    def set_dt(self, dt=None):
        """Set sampling time of the block."""
        if self._initialized:
            raise SetupError('the block has been initialized and cannot '
                             'change sampling time')
        self._dt = dt

    # called by self.initialize
    def _set_stepRate(self):
        """Set step rea"""
        stepRate = self.dt / self.system.dt
        if abs(stepRate / round(stepRate) - 1) <= STEPRATETOL:
            self._stepRate = round(stepRate)
        else:
            raise InitializationError('sample time of "{name}" must be an '
                                      'interger multiple '
                                      'of system'.format(name=self))

    def reset(self):
        """Reset the block to initial state."""
        self._n = None
        self._n_block = None
        self._initialized = False
        for signal in self.inports + self.outports:
            signal.reset()

    def _assert_initialization(self):
        try:
            assert(self._initialized)
        except AssertionError:
            raise InitializationError('{name} has not been initialized'
                                      .format(name=self))

    def initialize(self):
        """Initialize the block.
        0. check whether the block has been initialized
        1. set up basic variables
        2. call initialization function of the block
        3. set up remaining variables
        """
        if self._initialized:
            raise InitializationError('{name} has been initialized'
                                      .format(name=self))
        if self._system is None:
            raise InitializationError('"{name}" does not belong to any system'
                                      .format(name=self))
        self._set_stepRate()
        self.INITFUNC()
        self._initialized = True
        self._n = -1
        self._n_block = -1

    def step(self):
        """Block step.
        0. check whether has stepped in this time step
        1. check the sample rate to determine whether to step
        2. output step
        2. refresh input signal and block step
        3. increase step n
        """
        self._assert_initialization()
        sn = self.system.n

        if sn is None:
            raise StepError('system of {b} has not been initialized'
                            .format(s=self.system, b=self))

        # do nothing if this time step has been updated
        if self._n == sn:
            return

        # whether sampling time is reached
        if (self._n_block + 1) * self._stepRate == sn:

            # always perform output_step before _block_step
            for i in range(len(self._outports)):
                # do output_step if it is not done in this step
                if self._outports[i]._n < sn:
                    self.output_step(i)

            self._n_block += 1
            self._n += 1        # now, self._n == self.system.n
            self._block_step()

            # check outports
            for p in self._outports:
                if p._n != sn:
                    raise StepError("step of {s} at n = {n} failed because at "
                                    "least one of its outports is not updated"
                                    .format(s=self, n=sn))

        # always make sure self._n == self.system.n
        else:
            self._n += 1

    def output_step(self, outportId):
        """Update the output signal of the port before this step."""
        outport = self._outports[outportId]
        sn = self.system.n
        # whether sampling time is reached
        if (self._n_block + 1) * self._stepRate == sn:
            y = self.OUTPUTSTEP(outportId)
            if y is not NA:
                outport.set(y, sn)
        else:
            outport._n = sn

    def _block_step(self):
        """The function of the block."""
        xs = []
        for port in self._inports:
            port._step()
            value, n = port.get()
            if n != self.n:
                raise StepError('Step of {name} failed'.format(name=port))
            xs.append(value)

        ys = self.BLOCKSTEP(*xs)

        try:
            if len(ys) != len(self._outports):
                raise DefinitionError("length of output from BLOCKSTEP does not "
                                      "match its number of outports of {s}"
                                      .format(s=self))
        except TypeError:
            raise DefinitionError("output of {s} should be iterable with size "
                                  "{size}"
                                  .format(s=self,
                                          size=len(self._outports)))

        for i, y in enumerate(ys):
            if y is NA:
                continue
            self._outports[i].set(y, self.n)

    # The following functions can be overwritten by children

    def INITFUNC(self):
        """The method for initializing the block.

        This function is invoked once after the simulation system is
        initialized. It is usually used to set up the initial values for the
        simulation.
        """
        pass

    def OUTPUTSTEP(self, portid):
        """Method for giving outputs indenpendent of the inputs of this time
        step.

        OUTPUTSTEP is invoked as many times as number of outports of the block
        at the beginning of each iteration during the simulation. This function
        provides the outputs of the block which is independent of the inputs to
        the block at this time step. OUTPUTSTEP takes  exactly one argument,
        which is an interger indicating the portid of the output. It should
        return the corresponding output for this time step if it is independent
        of the input. However, if the output of the corresponding port is
        dependent on the input, this function should return sim.NA for the
        port.
        """
        return NA

    def BLOCKSTEP(self, *xs):
        """ Method for giving outputs dependent on the inputs of this time
        step.

        The blockstep is called once in each iteration after all the inputs to
        this block are known, and it should provides output for all the ports
        unless the corresponding port has already been set by outputstep. This
        function should take the same number input arguments as that defined
        by nin, which are the inputs to the block, and returns an iterable
        object with length nout. Each element of the iterable object is the
        output of the corresponding port. If the output of the port has already
        been given by outputstep, it may be set to sim.NA.
        """
        return [NA] * len(self._outports)
