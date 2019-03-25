"""user_defined_function.py

This module contains the interface for user-defined functions.
"""


from ..simsys import BaseBlock


__all__ = ['UserDefinedFunction', 'PythonFunction']


class UserDefinedFunction(BaseBlock):
    """The UserDefinedFunction block.

    This block runs the user defined function in each iteration.

    Parameters
    ----------
    func: callable
        The user defined function processing the input signals. The number of 
        inputs to the function is defined by nin. The output of the function 
        can be iterable and the length should be given by nout, and each
        element in the iterable object will be sent to the outports. However,
        if nout is set to None, the output of the function is not necessarily
        iterable and it will be sent to outports[0].

    t_as_arg: bool, optional
        Whether the user-defined function `func` accepts `t` as a keyword
        argument. If it is True, the simulation time will be an additional
        keyword argument passed to the function within each iteration. (default
        to False)

    nin: int, optional
        Number of input signals. This should also be the number of input 
        parameters of func. (default to 1)

    nout: int or None, optional
        Number of output signals. If nout is None, output of the user-defined
        function will be sent to outports[0].
        (default = None)

    dt: float, optional
        Sampling time of this block. The input and output be evaluate once
        for each sampling time.

    name: string, optional
        The name of this block.
    """

    def __init__(self, func, t_as_arg=False, nin=1, nout=None, dt=None,
                 name='UserDefinedFunction'):
        if nout is None:
            super().__init__(nin=nin, nout=1, dt=dt, name=name)
        else:
            super().__init__(nin=nin, nout=nout, dt=dt, name=name)

        if t_as_arg:
            self._func = lambda _t, *_xs: func(*_xs, t=_t)
        else:
            self._func = lambda _t, *_xs: func(*_xs)
        self._direct_output = False if nout is None else True

    def BLOCKSTEP(self, *xs):
        if self._direct_output:
            return self._func(self.t, *xs)
        else:
            return self._func(self.t, *xs),


class PythonFunction(BaseBlock):
    """Provides python functions to act as a block.

    It is a class provides interface for users to highly customize their
    blocks. Users may define their routine either during initializing or
    runtime. Users may also writting their own classes by directly inheriting
    BaseBlock.


    Parameters
    ----------
    nin: int, optional
        Number of inputs of this block.

    nout: int, optional
        Number of outputs of this block.

    initfunc: callable or None, optional
        The user-defined function for INITFUNC. The initfunc should take no
        input arguments and returns nothing (though it is not mandatory). This
        function is invoked once the simulation system is initialized. It is 
        usually used to set up some variables for the simulation. If initfunc
        is set to None, the defeault version in BaseBlock will be used.
        (default to None)

    outputstep: callable or None, optional
        The user-defined function for OUTPUTSTEP. The outputstep is invoked as
        many times as nout at the beginning of each iteration during the
        simulation. This function provides the outputs of the block which is
        independent of the inputs to the block at this time step. The
        outputstep should take  exactly one input argument, which is an
        interger indicates the portid of the output. It should returns the
        corresponding output for this time step if it is independent of the
        input. However, if the output of the corresponding port is dependent on
        the input, this function should return sim.NA for this port. If it is
        set to be None, the default version in BaseBlock will be used. (default
        to None)

    blockstep: callable or None, optional
        The user-defined function for BLOCKSTEP. The blockstep is called once
        in each iteration, and it should provides output for all the ports
        unless the corresponding port has already been set by outputstep. This
        function should take the same number input arguments as that defined
        by nin, which are the inputs to the block, and returns a tuple with
        length nout. Each element of the tuple is the output of the
        corresponding port. If the output of the port has already been given
        in outputstep, it may be set to sim.NA. if blockstep is set to None,
        the default version in BaseBlock will be used. (default to None)

    dt: float, optional
        Sampling time of this block. The input and output be evaluate once
        for each sampling time.

    name: string, optional
        The name of this block.
    """

    def __init__(self, nin=1, nout=1, initfunc=None, outputstep=None,
                 blockstep=None, dt=None, name='PythonFunction'):
        super().__init__(nin=nin, nout=nout, dt=dt, name=name)

        self.set_initfunc(initfunc)
        self.set_outputstep(outputstep)
        self.set_blockstep(blockstep)

    def set_initfunc(self, initfunc):
        """Set INITFUNC for the block."""
        self._initfunc = initfunc

    def set_outputstep(self, outputstep):
        """Set OUTPUTSTEP for the block."""
        self._outputstep = outputstep

    def set_blockstep(self, blockstep):
        """Set BLOCKSTEP for the block."""
        self._blockstep = blockstep

    def INITFUNC(self):
        if self._initfunc is not None:
            return self._initfunc()
        else:
            return super().INITFUNC()

    def OUTPUTSTEP(self, portid):
        if self._outputstep is not None:
            return self._outputstep(portid)
        else:
            return super().OUTPUTSTEP(portid)

    def BLOCKSTEP(self, *xs):
        if self._blockstep is not None:
            return self._blockstep(*xs)
        else:
            return super().BLOCKSTEP(*xs)
