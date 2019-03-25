"""user_defined_function.py

This module contains the interface for user-defined functions.
"""


from ..simsys import BaseBlock


__all__ = ['UserDefinedFunction']


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


